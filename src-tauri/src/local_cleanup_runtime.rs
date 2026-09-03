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
    if provider.id != LOCAL_CLEANUP_PROVIDER_ID {
        return Ok(None);
    }

    let request_guard = Arc::clone(&supervisor().request_gate).lock_owned().await;
    supervisor().ensure_ready(provider).await?;
    Ok(Some(LocalRuntimeLease {
        _request_guard: request_guard,
        finished: false,
    }))
}

pub struct LocalRuntimeLease {
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
        supervisor().stop_child("cleanup request ended before successful completion");
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
    let deadline = Instant::now() + LOCAL_RUNTIME_STARTUP_TIMEOUT;
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
                LOCAL_RUNTIME_STARTUP_TIMEOUT
            ));
        }
        tokio::time::sleep(LOCAL_RUNTIME_HEALTH_POLL).await;
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
