use crate::settings::{PostProcessProvider, LOCAL_CLEANUP_PROVIDER_ID};
use log::{debug, info, warn};
use std::env;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

const LOCAL_RUNTIME_COMMAND_ENV: &str = "HANDY_LOCAL_CLEANUP_COMMAND";
const LOCAL_RUNTIME_ARGS_ENV: &str = "HANDY_LOCAL_CLEANUP_ARGS";
const LOCAL_RUNTIME_CPU_ARGS_ENV: &str = "HANDY_LOCAL_CLEANUP_CPU_ARGS";
const LOCAL_RUNTIME_STARTUP_TIMEOUT: Duration = Duration::from_secs(8);
const LOCAL_RUNTIME_HEALTH_TIMEOUT: Duration = Duration::from_millis(500);
const LOCAL_RUNTIME_HEALTH_POLL: Duration = Duration::from_millis(100);

/// Backend cleanup policy used by the Wave 1 runtime boundary. The settings/UI
/// steward can map its persisted mode onto this type without giving `off` or
/// deterministic `fast` cleanup any path that can warm the local model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CleanupExecutionMode {
    #[allow(dead_code)] // Explicit non-runtime policy state retained for the settings boundary.
    Off,
    #[allow(dead_code)] // Explicit deterministic-cleanup state must never warm the local model.
    Fast,
    LocalAi,
}

impl CleanupExecutionMode {
    fn requires_local_runtime(self) -> bool {
        matches!(self, Self::LocalAi)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LaunchAttempt {
    command: String,
    args: Vec<String>,
    label: &'static str,
}

struct ManagedChild {
    child: Child,
    label: &'static str,
    generation: u64,
}

#[derive(Default)]
struct RuntimeState {
    child: Option<ManagedChild>,
}

impl RuntimeState {
    fn reap_exited(&mut self) -> Result<bool, String> {
        let Some(managed) = self.child.as_mut() else {
            return Ok(false);
        };

        match managed.child.try_wait() {
            Ok(Some(status)) => {
                warn!(
                    "Local cleanup runtime generation {} ({}) exited unexpectedly with status {}",
                    managed.generation, managed.label, status
                );
                self.child = None;
                Ok(true)
            }
            Ok(None) => Ok(false),
            Err(err) => {
                let generation = managed.generation;
                let label = managed.label;
                self.stop_child("failed to inspect runtime process");
                Err(format!(
                    "Failed to inspect local cleanup runtime generation {generation} ({label}): {err}"
                ))
            }
        }
    }

    fn stop_child(&mut self, reason: &str) {
        let Some(mut managed) = self.child.take() else {
            return;
        };

        debug!(
            "Stopping local cleanup runtime generation {} ({}) after {}",
            managed.generation, managed.label, reason
        );
        match managed.child.try_wait() {
            Ok(Some(_)) => return,
            Ok(None) => {}
            Err(err) => {
                warn!(
                    "Failed to inspect local cleanup runtime generation {} before shutdown: {}",
                    managed.generation, err
                );
            }
        }

        if let Err(err) = managed.child.kill() {
            warn!(
                "Failed to kill local cleanup runtime generation {}: {}",
                managed.generation, err
            );
        }
        if let Err(err) = managed.child.wait() {
            warn!(
                "Failed to reap local cleanup runtime generation {}: {}",
                managed.generation, err
            );
        }
    }
}

struct RuntimeSupervisor {
    state: Mutex<RuntimeState>,
    lifecycle_gate: AsyncMutex<()>,
    request_gate: Arc<AsyncMutex<()>>,
    next_generation: AtomicU64,
}

struct StartupChildGuard<'a> {
    supervisor: &'a RuntimeSupervisor,
    armed: bool,
}

impl Drop for StartupChildGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            // `ensure_ready` is async. If its future is cancelled while a child
            // is in the startup/health loop, clean that child up immediately.
            self.supervisor
                .stop_child("cleanup runtime startup was cancelled");
        }
    }
}

impl RuntimeSupervisor {
    fn new() -> Self {
        Self {
            state: Mutex::new(RuntimeState::default()),
            lifecycle_gate: AsyncMutex::new(()),
            request_gate: Arc::new(AsyncMutex::new(())),
            next_generation: AtomicU64::new(1),
        }
    }

    fn lock_state(&self) -> MutexGuard<'_, RuntimeState> {
        self.state.lock().unwrap_or_else(|poisoned| {
            warn!("Recovered poisoned local cleanup runtime supervisor state");
            poisoned.into_inner()
        })
    }

    async fn ensure_ready(&self, provider: &PostProcessProvider) -> Result<(), String> {
        let _lifecycle = self.lifecycle_gate.lock().await;

        {
            let mut state = self.lock_state();
            let _ = state.reap_exited()?;
        }

        if health_check(provider).await {
            return Ok(());
        }

        // A child that is alive but no longer healthy must not be left resident.
        // Kill/reap it before trying a fresh accelerated or CPU launch.
        self.stop_child("health check failed");

        let attempts = configured_launch_attempts()?;
        let mut failures = Vec::new();
        for attempt in attempts {
            let generation = self.next_generation.fetch_add(1, Ordering::Relaxed);
            let mut startup_guard = StartupChildGuard {
                supervisor: self,
                armed: false,
            };
            match spawn_attempt(&attempt, generation) {
                Ok(managed) => {
                    info!(
                        "Started supervised local cleanup runtime generation {} ({})",
                        generation, attempt.label
                    );
                    self.lock_state().child = Some(managed);
                    startup_guard.armed = true;
                }
                Err(err) => {
                    failures.push(format!("{} launch failed: {err}", attempt.label));
                    continue;
                }
            }

            match wait_until_healthy(self, provider, generation).await {
                Ok(()) => {
                    startup_guard.armed = false;
                    return Ok(());
                }
                Err(err) => {
                    failures.push(format!("{} startup failed: {err}", attempt.label));
                    self.stop_child("bounded startup failed");
                    startup_guard.armed = false;
                }
            }
        }

        Err(format!(
            "Local cleanup runtime could not become healthy: {}",
            failures.join("; ")
        ))
    }

    fn stop_child(&self, reason: &str) {
        self.lock_state().stop_child(reason);
    }
}

fn supervisor() -> &'static RuntimeSupervisor {
    static SUPERVISOR: OnceLock<RuntimeSupervisor> = OnceLock::new();
    SUPERVISOR.get_or_init(RuntimeSupervisor::new)
}

/// Lease the resident local cleanup runtime for one request.
///
/// Requests are serialized because llama.cpp-compatible local runtimes are
/// commonly configured for one decode at a time. The child itself stays resident
/// after a successful request; dropping an unfinished lease (for example because
/// the caller cancelled the async request) kills and reaps the managed child so
/// abandoned inference cannot remain stranded in the background.
pub async fn acquire(provider: &PostProcessProvider) -> Result<Option<LocalRuntimeLease>, String> {
    acquire_for_mode(CleanupExecutionMode::LocalAi, provider).await
}

/// Acquire the resident runtime only for local-AI cleanup. Off mode is the raw
/// deterministic transcript and Fast mode is code-only cleanup; neither may
/// probe, launch, or warm a local model process.
pub(crate) async fn acquire_for_mode(
    mode: CleanupExecutionMode,
    provider: &PostProcessProvider,
) -> Result<Option<LocalRuntimeLease>, String> {
    if !mode.requires_local_runtime() || provider.id != LOCAL_CLEANUP_PROVIDER_ID {
        return Ok(None);
    }

    let runtime = supervisor();
    let request_guard = Arc::clone(&runtime.request_gate).lock_owned().await;
    runtime.ensure_ready(provider).await?;
    Ok(Some(LocalRuntimeLease {
        supervisor: runtime,
        _request_guard: request_guard,
        finished: false,
    }))
}

pub struct LocalRuntimeLease {
    supervisor: &'static RuntimeSupervisor,
    _request_guard: OwnedMutexGuard<()>,
    finished: bool,
}

impl LocalRuntimeLease {
    /// Mark a request as normally completed. The resident child remains warm.
    pub fn complete(mut self) {
        self.finished = true;
    }
}

impl Drop for LocalRuntimeLease {
    fn drop(&mut self) {
        if self.finished {
            return;
        }

        // This covers request cancellation, timeout, transport failure, malformed
        // HTTP response, and panics unwinding through the request future. A local
        // cleanup failure is fail-open at the caller, while the abandoned child is
        // never left doing work that nobody will consume.
        self.supervisor
            .stop_child("cleanup request ended before successful completion");
    }
}

/// Explicit application-shutdown hook. Static supervisors do not have a useful
/// Drop point, so the Tauri exit event calls this to kill and reap the child.
pub fn shutdown() {
    supervisor().stop_child("application shutdown");
}

async fn wait_until_healthy(
    supervisor: &RuntimeSupervisor,
    provider: &PostProcessProvider,
    generation: u64,
) -> Result<(), String> {
    wait_until_healthy_with_timing(
        supervisor,
        provider,
        generation,
        LOCAL_RUNTIME_STARTUP_TIMEOUT,
        LOCAL_RUNTIME_HEALTH_POLL,
    )
    .await
}

async fn wait_until_healthy_with_timing(
    supervisor: &RuntimeSupervisor,
    provider: &PostProcessProvider,
    generation: u64,
    startup_timeout: Duration,
    health_poll: Duration,
) -> Result<(), String> {
    let deadline = Instant::now() + startup_timeout;
    loop {
        {
            let mut state = supervisor.lock_state();
            if state.reap_exited()? {
                return Err(format!(
                    "runtime generation {generation} exited before its health check succeeded"
                ));
            }
        }

        if health_check(provider).await {
            info!(
                "Local cleanup runtime generation {} passed its health check",
                generation
            );
            return Ok(());
        }

        if Instant::now() >= deadline {
            return Err(format!(
                "timed out after {:?} waiting for the local cleanup runtime health endpoint",
                startup_timeout
            ));
        }
        tokio::time::sleep(health_poll).await;
    }
}

async fn health_check(provider: &PostProcessProvider) -> bool {
    let url = format!("{}/models", provider.base_url.trim_end_matches('/'));
    let client = match reqwest::Client::builder()
        .timeout(LOCAL_RUNTIME_HEALTH_TIMEOUT)
        .build()
    {
        Ok(client) => client,
        Err(err) => {
            warn!("Failed to build local cleanup health-check client: {err}");
            return false;
        }
    };

    match client.get(url).send().await {
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    }
}

fn spawn_attempt(attempt: &LaunchAttempt, generation: u64) -> Result<ManagedChild, String> {
    let child = Command::new(&attempt.command)
        .args(&attempt.args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|err| {
            format!(
                "failed to spawn configured local cleanup runtime '{}': {err}",
                attempt.command
            )
        })?;

    Ok(ManagedChild {
        child,
        label: attempt.label,
        generation,
    })
}

fn configured_launch_attempts() -> Result<Vec<LaunchAttempt>, String> {
    let command = env::var(LOCAL_RUNTIME_COMMAND_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            format!(
                "Local cleanup endpoint is unavailable and {LOCAL_RUNTIME_COMMAND_ENV} is not configured"
            )
        })?;
    let primary_args = parse_args_env(LOCAL_RUNTIME_ARGS_ENV)?.unwrap_or_default();
    let mut attempts = vec![LaunchAttempt {
        command: command.clone(),
        args: primary_args.clone(),
        label: "configured",
    }];

    let cpu_args = match parse_args_env(LOCAL_RUNTIME_CPU_ARGS_ENV)? {
        Some(args) => Some(args),
        None => derive_llama_cpu_args(&primary_args),
    };
    if let Some(cpu_args) = cpu_args.filter(|args| *args != primary_args) {
        attempts.push(LaunchAttempt {
            command,
            args: cpu_args,
            label: "cpu-fallback",
        });
    }

    Ok(attempts)
}

fn parse_args_env(name: &str) -> Result<Option<Vec<String>>, String> {
    let Some(raw) = env::var(name).ok().filter(|value| !value.trim().is_empty()) else {
        return Ok(None);
    };

    serde_json::from_str::<Vec<String>>(&raw)
        .map(Some)
        .map_err(|err| format!("{name} must be a JSON array of arguments: {err}"))
}

/// llama.cpp exposes GPU offload as `-ngl` / `--n-gpu-layers`. When the
/// configured command uses either conventional form, a failed accelerated
/// startup can retry the exact same resident runtime with zero GPU layers. For
/// other runtimes callers can provide an explicit HANDY_LOCAL_CLEANUP_CPU_ARGS
/// JSON array; if neither applies, no unsupported fallback is invented.
fn derive_llama_cpu_args(primary: &[String]) -> Option<Vec<String>> {
    let mut cpu = primary.to_vec();
    let mut replaced = false;
    let mut index = 0;
    while index < cpu.len() {
        if matches!(cpu[index].as_str(), "-ngl" | "--n-gpu-layers") && index + 1 < cpu.len() {
            cpu[index + 1] = "0".to_string();
            replaced = true;
            index += 2;
            continue;
        }
        if let Some((flag, _)) = cpu[index].split_once('=') {
            if matches!(flag, "--n-gpu-layers") {
                cpu[index] = "--n-gpu-layers=0".to_string();
                replaced = true;
            }
        }
        index += 1;
    }

    replaced.then_some(cpu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn serve_health_checks(count: usize) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind health fixture");
        let address = listener.local_addr().expect("health fixture address");
        let handle = tokio::spawn(async move {
            for _ in 0..count {
                let (mut stream, _) = listener.accept().await.expect("accept health check");
                let mut request = [0_u8; 1024];
                let _ = stream.read(&mut request).await.expect("read health check");
                stream
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\nConnection: close\r\n\r\n{\"data\":[]}",
                    )
                    .await
                    .expect("write health response");
            }
        });
        (format!("http://{address}"), handle)
    }

    const TEST_CHILD_MODE_ENV: &str = "HANDY_LOCAL_CLEANUP_TEST_CHILD_MODE";

    fn test_provider(base_url: &str) -> PostProcessProvider {
        PostProcessProvider {
            id: LOCAL_CLEANUP_PROVIDER_ID.to_string(),
            label: "Local cleanup test".to_string(),
            base_url: base_url.to_string(),
            allow_base_url_edit: false,
            models_endpoint: None,
            supports_structured_output: false,
        }
    }

    fn spawn_fixture_child(mode: &str) -> Child {
        Command::new(std::env::current_exe().expect("test executable path"))
            .arg("local_cleanup_child_fixture")
            .arg("--nocapture")
            .env(TEST_CHILD_MODE_ENV, mode)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn local cleanup fixture child")
    }

    #[test]
    fn local_cleanup_child_fixture() {
        match std::env::var(TEST_CHILD_MODE_ENV).as_deref() {
            Ok("wait") => thread::sleep(Duration::from_secs(30)),
            Ok("exit") => {}
            _ => {}
        }
    }

    #[test]
    fn cleanup_modes_warm_runtime_only_for_local_ai() {
        assert!(!CleanupExecutionMode::Off.requires_local_runtime());
        assert!(!CleanupExecutionMode::Fast.requires_local_runtime());
        assert!(CleanupExecutionMode::LocalAi.requires_local_runtime());
    }

    #[tokio::test]
    async fn off_and_fast_modes_do_not_probe_or_launch_local_runtime() {
        // This endpoint is deliberately unreachable. Returning successfully for
        // Off/Fast proves mode gating happens before health checks or configured
        // process discovery.
        let provider = test_provider("http://127.0.0.1:1");
        assert!(acquire_for_mode(CleanupExecutionMode::Off, &provider)
            .await
            .expect("off mode")
            .is_none());
        assert!(acquire_for_mode(CleanupExecutionMode::Fast, &provider)
            .await
            .expect("fast mode")
            .is_none());
    }

    #[tokio::test]
    async fn healthy_resident_runtime_is_reused_across_cleanup_requests() {
        let (base_url, server) = serve_health_checks(2).await;
        let provider = test_provider(&base_url);
        let runtime = RuntimeSupervisor::new();
        let child = spawn_fixture_child("wait");
        let original_pid = child.id();
        runtime.lock_state().child = Some(ManagedChild {
            child,
            label: "test-resident",
            generation: 42,
        });

        runtime
            .ensure_ready(&provider)
            .await
            .expect("first request ready");
        let first_pid = runtime
            .lock_state()
            .child
            .as_ref()
            .expect("resident child retained")
            .child
            .id();
        runtime
            .ensure_ready(&provider)
            .await
            .expect("second request ready");
        let second_pid = runtime
            .lock_state()
            .child
            .as_ref()
            .expect("resident child reused")
            .child
            .id();

        assert_eq!(first_pid, original_pid);
        assert_eq!(second_pid, original_pid);
        runtime.stop_child("resident reuse test complete");
        server.await.expect("health fixture completed");
    }

    #[tokio::test]
    async fn cancelled_request_lease_kills_child_and_allows_clean_recovery() {
        let runtime: &'static RuntimeSupervisor = Box::leak(Box::new(RuntimeSupervisor::new()));
        runtime.lock_state().child = Some(ManagedChild {
            child: spawn_fixture_child("wait"),
            label: "cancelled-request",
            generation: 51,
        });
        let request_guard = Arc::clone(&runtime.request_gate).lock_owned().await;
        let cancelled_lease = LocalRuntimeLease {
            supervisor: runtime,
            _request_guard: request_guard,
            finished: false,
        };

        drop(cancelled_lease);
        assert!(runtime.lock_state().child.is_none());

        // The supervisor remains reusable after cancellation; a later request
        // can own a fresh child rather than inheriting abandoned inference.
        runtime.lock_state().child = Some(ManagedChild {
            child: spawn_fixture_child("wait"),
            label: "recovered-request",
            generation: 52,
        });
        assert_eq!(
            runtime
                .lock_state()
                .child
                .as_ref()
                .expect("replacement child")
                .generation,
            52
        );
        runtime.stop_child("cancellation recovery test complete");
    }

    #[test]
    fn bounded_startup_health_wait_times_out() {
        let supervisor = RuntimeSupervisor::new();
        let provider = test_provider("http://127.0.0.1:1");
        let started = Instant::now();
        let result = tauri::async_runtime::block_on(wait_until_healthy_with_timing(
            &supervisor,
            &provider,
            1,
            Duration::from_millis(30),
            Duration::from_millis(1),
        ));

        let error = result.expect_err("unhealthy runtime must time out");
        assert!(error.contains("timed out after"));
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn startup_cancellation_guard_kills_and_reaps_managed_child() {
        let supervisor = RuntimeSupervisor::new();
        let child = spawn_fixture_child("wait");
        supervisor.lock_state().child = Some(ManagedChild {
            child,
            label: "test",
            generation: 7,
        });

        let started = Instant::now();
        {
            let _guard = StartupChildGuard {
                supervisor: &supervisor,
                armed: true,
            };
        }

        assert!(supervisor.lock_state().child.is_none());
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn exited_runtime_is_reaped_before_reuse() {
        let mut child = spawn_fixture_child("exit");
        let deadline = Instant::now() + Duration::from_secs(2);
        while child.try_wait().expect("inspect fixture child").is_none() {
            assert!(Instant::now() < deadline, "fixture child did not exit");
            thread::sleep(Duration::from_millis(5));
        }

        let mut state = RuntimeState {
            child: Some(ManagedChild {
                child,
                label: "test",
                generation: 9,
            }),
        };
        assert!(state.reap_exited().expect("reap exited runtime"));
        assert!(state.child.is_none());
    }

    #[test]
    fn derives_cpu_fallback_for_common_llama_gpu_flags() {
        assert_eq!(
            derive_llama_cpu_args(&[
                "-m".into(),
                "cleanup.gguf".into(),
                "-ngl".into(),
                "99".into(),
            ]),
            Some(vec![
                "-m".into(),
                "cleanup.gguf".into(),
                "-ngl".into(),
                "0".into(),
            ])
        );
        assert_eq!(
            derive_llama_cpu_args(&["--n-gpu-layers=all".into()]),
            Some(vec!["--n-gpu-layers=0".into()])
        );
    }

    #[test]
    fn does_not_invent_cpu_flags_for_unknown_runtimes() {
        assert_eq!(
            derive_llama_cpu_args(&["--model".into(), "cleanup.gguf".into()]),
            None
        );
    }

    #[test]
    fn launch_attempts_keep_cpu_fallback_after_primary() {
        let primary = vec!["-ngl".to_string(), "99".to_string()];
        let cpu = derive_llama_cpu_args(&primary).unwrap();
        assert_eq!(cpu, vec!["-ngl", "0"]);
        assert_ne!(cpu, primary);
    }
}
