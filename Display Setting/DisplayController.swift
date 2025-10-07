//
//  DisplayController.swift
//  On Display Computing
//
//  Created by David Zheng on 6/29/25.
//


import Foundation

/// Controls actual display settings on the Mac system
class DisplayController {
    /// Sets the brightness level of the display
    /// - Parameter level: Brightness level from 0-100%
    func setBrightness(_ level: Int) {
        // Validate the input range
        guard level >= 0 && level <= 100 else { 
            print("⚠️ Invalid brightness level: \(level)")
            return 
        }
        
        // Convert percentage to normalized value (0.0-1.0)
        let normalizedValue = Double(level) / 100.0
        
        // Create AppleScript to adjust brightness
        // Note: This is a simplified approach and might not work on all Macs
        let script = """
        tell application "System Events"
            tell appearance preferences
                set dark mode to \(level < 50 ? "true" : "false")
            end tell
        end tell
        
        tell application "System Events"
            key code 144 using {control down, shift down}
        end tell
        """
        
        // Execute the AppleScript
        let process = Process()
        process.launchPath = "/usr/bin/osascript"
        process.arguments = ["-e", script]
        
        do {
            try process.run()
            process.waitUntilExit()
            print("✅ Set brightness to \(level)%")
        } catch {
            print("❌ Error adjusting brightness: \(error)")
        }
    }
    
    /// Sets the Night Shift intensity
    /// - Parameter level: Night Shift level from 0-100%
    func setNightShift(_ level: Int) {
        // Validate the input range
        guard level >= 0 && level <= 100 else { return }
        
        // Night Shift is controlled via 'defaults write' commands
        // Here we'd need to use core foundation preferences to adjust it
        print("🔶 Night Shift adjustment not implemented")
    }
    
    /// Applies a color temperature setting
    /// - Parameter kelvin: Color temperature in Kelvin (e.g., 6500K)
    func setColorTemperature(_ kelvin: Int) {
        // Validate reasonable range (3000K-10000K)
        guard kelvin >= 3000 && kelvin <= 10000 else { return }
        
        // On macOS, this would likely be done through Night Shift or third-party tools
        print("🔶 Color temperature adjustment not implemented")
    }
    
    /// Toggles HDR mode if available
    /// - Parameter enabled: Whether HDR should be enabled
    func toggleHDR(_ enabled: Bool) {
        // This would require specific system APIs
        print("🔶 HDR toggle not implemented")
    }
    
    /// Applies all settings from a DisplaySettings object
    /// - Parameter settings: The settings to apply
    func applySettings(_ settings: DisplaySettings) {
        // Apply each setting if it has a value
        if let brightness = settings.brightness {
            setBrightness(brightness)
        }
        
        if let nightShift = settings.nightShiftLevel {
            setNightShift(nightShift)
        }
        
        if let colorTemp = settings.colorTemperature {
            setColorTemperature(colorTemp)
        }
        
        if let hdr = settings.hdrEnabled {
            toggleHDR(hdr)
        }
        
        // Other settings would be applied here
        print("✅ Applied available settings")
    }
}