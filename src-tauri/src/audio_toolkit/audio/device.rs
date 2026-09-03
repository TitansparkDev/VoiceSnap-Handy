use cpal::traits::{DeviceTrait, HostTrait};

pub struct CpalDeviceInfo {
    pub index: String,
    /// Stable CPAL device identity when the active host exposes one. CPAL 0.17+
    /// defines this identifier for persistence across process restarts and
    /// reconnects; if a backend cannot provide it we retain strict name matching.
    pub stable_id: Option<String>,
    pub name: String,
    pub is_default: bool,
    pub device: cpal::Device,
}

pub fn list_input_devices() -> Result<Vec<CpalDeviceInfo>, Box<dyn std::error::Error>> {
    let host = crate::audio_toolkit::get_cpal_host();
    let default_name = host
        .default_input_device()
        .and_then(|d| d.description().ok().map(|desc| desc.name().to_string()));

    let mut out = Vec::<CpalDeviceInfo>::new();

    for (index, device) in host.input_devices()?.enumerate() {
        let name = device
            .description()
            .map(|desc| desc.name().to_string())
            .unwrap_or_else(|_| "Unknown".into());

        let is_default = Some(name.clone()) == default_name;

        let stable_id = device.id().ok().map(|id| id.to_string());

        out.push(CpalDeviceInfo {
            index: index.to_string(),
            stable_id,
            name,
            is_default,
            device,
        });
    }

    Ok(out)
}

pub fn list_output_devices() -> Result<Vec<CpalDeviceInfo>, Box<dyn std::error::Error>> {
    let host = crate::audio_toolkit::get_cpal_host();
    let default_name = host
        .default_output_device()
        .and_then(|d| d.description().ok().map(|desc| desc.name().to_string()));

    let mut out = Vec::<CpalDeviceInfo>::new();

    for (index, device) in host.output_devices()?.enumerate() {
        let name = device
            .description()
            .map(|desc| desc.name().to_string())
            .unwrap_or_else(|_| "Unknown".into());

        let is_default = Some(name.clone()) == default_name;

        let stable_id = device.id().ok().map(|id| id.to_string());

        out.push(CpalDeviceInfo {
            index: index.to_string(),
            stable_id,
            name,
            is_default,
            device,
        });
    }

    Ok(out)
}
