//
//  DisplaySettings.swift
//  On Display Computing
//
//  Created by David Zheng on 6/29/25.
//


import Foundation

/// Data structure to represent display settings recommendations
struct DisplaySettings: Codable {
    /// Screen brightness level (0-100%)
    var brightness: Int?
    
    /// Display contrast level (0-100%)
    var contrast: Int?
    
    /// Color temperature in Kelvin (e.g., 6500K)
    var colorTemperature: Int?
    
    /// Night Shift intensity level (0-100%)
    var nightShiftLevel: Int?
    
    /// True Black level for OLED displays (0-100%)
    var trueBlackLevel: Int?
    
    /// Gamma correction value (typically 1.8-2.2)
    var gamma: Float?
    
    /// Blue light filter intensity (0-100%)
    var blueLight: Int?
    
    /// Text sharpness enhancement level (0-100%)
    var textSharpness: Int?
    
    /// Response time setting for gaming (in ms)
    var responseTime: Int?
    
    /// HDR status (on/off)
    var hdrEnabled: Bool?
    
    /// Parses LLM output text into structured display settings
    /// - Parameter output: Raw text from the LLM containing settings recommendations
    /// - Returns: Structured DisplaySettings object
    static func fromLLMOutput(_ output: String) -> DisplaySettings {
        // Create an empty settings object to populate
        var settings = DisplaySettings()
        
        // Parse brightness using regular expression
        if let range = output.range(of: "Brightness: \\d+", options: .regularExpression) {
            // Extract just the numeric value
            let value = output[range].components(separatedBy: CharacterSet.decimalDigits.inverted).joined()
            settings.brightness = Int(value)
        }
        
        // Parse contrast setting
        if let range = output.range(of: "Contrast: \\d+", options: .regularExpression) {
            let value = output[range].components(separatedBy: CharacterSet.decimalDigits.inverted).joined()
            settings.contrast = Int(value)
        }
        
        // Parse color temperature (typically in Kelvin)
        if let range = output.range(of: "Color Temperature: \\d+", options: .regularExpression) {
            let value = output[range].components(separatedBy: CharacterSet.decimalDigits.inverted).joined()
            settings.colorTemperature = Int(value)
        }
        
        // Parse gamma value (floating point)
        if let range = output.range(of: "Gamma: \\d+\\.?\\d*", options: .regularExpression) {
            let valueStr = output[range].components(separatedBy: CharacterSet.decimalDigits.inverted.subtracting(CharacterSet(charactersIn: "."))).joined()
            settings.gamma = Float(valueStr)
        }
        
        // Parse blue light filter setting
        if let range = output.range(of: "Blue Light Filter: \\d+", options: .regularExpression) {
            let value = output[range].components(separatedBy: CharacterSet.decimalDigits.inverted).joined()
            settings.blueLight = Int(value)
        }
        
        // Parse text sharpness setting
        if let range = output.range(of: "Text Sharpness: \\d+", options: .regularExpression) {
            let value = output[range].components(separatedBy: CharacterSet.decimalDigits.inverted).joined()
            settings.textSharpness = Int(value)
        }
        
        // Parse night shift setting
        if let range = output.range(of: "Night Shift: \\d+", options: .regularExpression) {
            let value = output[range].components(separatedBy: CharacterSet.decimalDigits.inverted).joined()
            settings.nightShiftLevel = Int(value)
        }
        
        // Parse HDR setting (enabled/disabled)
        if let range = output.range(of: "HDR: (Enabled|Disabled)", options: .regularExpression) {
            let value = output[range].lowercased().contains("enabled")
            settings.hdrEnabled = value
        }
        
        // Parse response time (for gaming)
        if let range = output.range(of: "Response Time: \\d+", options: .regularExpression) {
            let value = output[range].components(separatedBy: CharacterSet.decimalDigits.inverted).joined()
            settings.responseTime = Int(value)
        }
        
        return settings
    }
}