//
//  DisplaySettingsDatabase.swift
//  On Display Computing
//
//  Created by David Zheng on 6/30/25.
//


import Foundation

/// Database of display settings for different application types
class DisplaySettingsDatabase {
    /// Singleton instance
    static let shared = DisplaySettingsDatabase()
    
    /// Knowledge base containing display settings for different app types
    private var knowledgeBase: [String: [String: Any]] = [:]
    
    /// Private initializer for singleton
    private init() {
        setupKnowledgeBase()
    }
    
    /// Sets up the knowledge base with recommended settings for different app types
    private func setupKnowledgeBase() {
        // Photo Editing Apps
        knowledgeBase["photo editing"] = [
            "description": "Applications for editing and processing photos and images",
            "brightness": 85,
            "contrast": 80,
            "color_temperature": 6500,
            "gamma": 2.2,
            "blue_light_filter": 10,
            "refresh_rate": 75,
            "rationale": "Photo editing requires accurate color representation and high brightness for detail visibility. A neutral color temperature of 6500K (daylight) ensures accurate color perception.",
            "examples": ["Adobe Photoshop", "Lightroom", "GIMP", "Pixelmator"]
        ]
        
        // Video Editing Apps
        knowledgeBase["video editing"] = [
            "description": "Applications for editing and processing video content",
            "brightness": 90,
            "contrast": 85,
            "color_temperature": 6000,
            "response_time": 5,
            "gamma": 2.4,
            "refresh_rate": 60,
            "HDR": true,
            "rationale": "Video editing benefits from high brightness and contrast for accurate color grading. HDR is recommended for editing HDR content.",
            "examples": ["Final Cut Pro", "Adobe Premiere", "DaVinci Resolve"]
        ]
        
        // Text Editing Apps
        knowledgeBase["text editing"] = [
            "description": "Applications for writing and editing text documents",
            "brightness": 65,
            "blue_light_filter": 40,
            "text_sharpness": 75,
            "color_temperature": 5500,
            "contrast": 65,
            "refresh_rate": 60,
            "rationale": "Text editing benefits from lower brightness and warmer color temperature to reduce eye strain during prolonged reading and writing sessions.",
            "examples": ["Microsoft Word", "Pages", "Google Docs", "TextEdit"]
        ]
        
        // Code Editing Apps
        knowledgeBase["code editor"] = [
            "description": "Applications for writing and editing programming code",
            "brightness": 60,
            "blue_light_filter": 50,
            "contrast": 70,
            "text_sharpness": 80,
            "color_temperature": 5000,
            "refresh_rate": 60,
            "rationale": "Programming involves long sessions of reading text, so settings focus on reducing eye strain with higher blue light filtering and warmer color temperature.",
            "examples": ["Visual Studio Code", "Xcode", "Sublime Text", "IntelliJ"]
        ]
        
        // Gaming Apps
        knowledgeBase["gaming"] = [
            "description": "Games and gaming platforms",
            "brightness": 90,
            "contrast": 85,
            "response_time": 1,
            "HDR": true,
            "gamma": 2.2,
            "color_temperature": 6500,
            "refresh_rate": 144,
            "rationale": "Gaming benefits from high brightness and contrast for better visibility. Fast response time and high refresh rates reduce motion blur in fast-paced games.",
            "examples": ["Steam games", "Epic Games", "Battle.net"]
        ]
        
        // Web Browsers
        knowledgeBase["web browser"] = [
            "description": "Web browsing applications",
            "brightness": 70,
            "blue_light_filter": 30,
            "contrast": 75,
            "color_temperature": 5800,
            "refresh_rate": 60,
            "rationale": "Web browsing involves reading various types of content, so settings balance readability with reduced eye strain.",
            "examples": ["Safari", "Chrome", "Firefox", "Edge"]
        ]
        
        // Terminal Apps
        knowledgeBase["terminal"] = [
            "description": "Command-line interface applications",
            "brightness": 55,
            "contrast": 80,
            "text_sharpness": 85,
            "blue_light_filter": 45,
            "color_temperature": 5000,
            "refresh_rate": 60,
            "rationale": "Terminal apps typically have text on dark backgrounds, so higher contrast improves readability while lower brightness reduces eye strain.",
            "examples": ["Terminal", "iTerm", "PowerShell"]
        ]
        
        // Streaming Apps
        knowledgeBase["streaming"] = [
            "description": "Video streaming applications",
            "brightness": 85,
            "contrast": 80,
            "color_temperature": 6000,
            "HDR": true,
            "gamma": 2.4,
            "refresh_rate": 60,
            "rationale": "Video streaming benefits from settings that enhance the viewing experience with higher brightness and contrast. HDR is beneficial for supported content.",
            "examples": ["Netflix", "YouTube", "Disney+", "Hulu"]
        ]
    }
    
    /// Gets settings for a specific app based on its type
    /// - Parameters:
    ///   - appName: Name of the application
    ///   - appType: Type/category of the application
    /// - Returns: Dictionary containing recommended settings
    func getSettings(for appName: String, appType: String) -> [String: Any]? {
        // Try to find exact match
        let normalizedType = appType.lowercased()
        
        if let settings = knowledgeBase[normalizedType] {
            return settings
        }
        
        // Try to find partial match
        for (knownType, settings) in knowledgeBase {
            if normalizedType.contains(knownType) || knownType.contains(normalizedType) {
                return settings
            }
            
            // Check if app name matches any examples
            if let examples = settings["examples"] as? [String] {
                for example in examples {
                    if appName.lowercased().contains(example.lowercased()) {
                        return settings
                    }
                }
            }
        }
        
        // Return general settings if no match found
        return knowledgeBase["web browser"]
    }
    
    /// Gets a formatted context for RAG based on the app type and name
    /// - Parameters:
    ///   - appName: Name of the application
    ///   - appType: Type/category of the application
    /// - Returns: String containing relevant context for the LLM
    func getRAGContext(for appName: String, appType: String) -> String {
        guard let settings = getSettings(for: appName, appType: appType) else {
            return "No specific display settings information available for \(appType) applications."
        }
        
        // Format the context for the LLM
        var context = "Display settings recommendations for \(appType) applications like \(appName):\n"
        
        if let description = settings["description"] as? String {
            context += "\nDescription: \(description)\n"
        }
        
        context += "\nRecommended settings:\n"
        
        if let brightness = settings["brightness"] as? Int {
            context += "- Brightness: \(brightness)%\n"
        }
        
        if let contrast = settings["contrast"] as? Int {
            context += "- Contrast: \(contrast)%\n"
        }
        
        if let colorTemp = settings["color_temperature"] as? Int {
            context += "- Color Temperature: \(colorTemp)K\n"
        }
        
        if let gamma = settings["gamma"] as? Double {
            context += "- Gamma: \(gamma)\n"
        }
        
        if let blueLight = settings["blue_light_filter"] as? Int {
            context += "- Blue Light Filter: \(blueLight)%\n"
        }
        
        if let textSharpness = settings["text_sharpness"] as? Int {
            context += "- Text Sharpness: \(textSharpness)%\n"
        }
        
        if let responseTime = settings["response_time"] as? Int {
            context += "- Response Time: \(responseTime)ms\n"
        }
        
        if let refreshRate = settings["refresh_rate"] as? Int {
            context += "- Refresh Rate: \(refreshRate)Hz\n"
        }
        
        if let hdr = settings["hdr"] as? Bool {
            context += "- HDR: \(hdr ? "Enabled" : "Disabled")\n"
        }
        
        if let rationale = settings["rationale"] as? String {
            context += "\nRationale: \(rationale)\n"
        }
        
        return context
    }
    
    /// Provides a specific setting value from the database
    /// - Parameters:
    ///   - settingName: Name of the setting to retrieve
    ///   - appName: Name of the application
    ///   - appType: Type/category of the application
    /// - Returns: String containing the setting value, or nil if not found
    func getSpecificSetting(name settingName: String, for appName: String, appType: String) -> String? {
        guard let settings = getSettings(for: appName, appType: appType) else { return nil }
        
        // Convert settingName to the key format used in our knowledge base
        let settingKey = settingName
            .lowercased()
            .replacingOccurrences(of: " ", with: "_")
        
        // Try to get the value
        if let value = settings[settingKey] {
            // Format the value appropriately
            if settingKey == "hdr" {
                return (value as? Bool == true) ? "ON" : "OFF"
            } else if let numValue = value as? Int {
                if ["brightness", "contrast", "blue_light_filter", "text_sharpness"].contains(settingKey) {
                    return "\(numValue)%"
                } else if settingKey == "color_temperature" {
                    return "\(numValue)K"
                } else if settingKey == "response_time" {
                    return "\(numValue)ms"
                } else if settingKey == "refresh_rate" {
                    return "\(numValue)Hz"
                } else {
                    return "\(numValue)"
                }
            } else if let doubleValue = value as? Double {
                return String(format: "%.1f", doubleValue)
            }
        }
        
        return nil
    }
    
    /// Formats settings as a clean string for display
    /// - Parameters:
    ///   - appName: Name of the application
    ///   - appType: Type/category of the application
    /// - Returns: Formatted settings string
    func getFormattedSettings(for appName: String, appType: String) -> String {
        guard let settings = getSettings(for: appName, appType: appType) else {
            return "No specific settings available for \(appType) applications."
        }
        
        var result = ""
        
        if let brightness = settings["brightness"] as? Int {
            result += "Brightness: \(brightness)%\n"
        }
        
        if let contrast = settings["contrast"] as? Int {
            result += "Contrast: \(contrast)%\n"
        }
        
        if let colorTemp = settings["color_temperature"] as? Int {
            result += "Color Temperature: \(colorTemp)K\n"
        }
        
        if let gamma = settings["gamma"] as? Double {
            result += "Gamma: \(gamma)\n"
        }
        
        if let blueLight = settings["blue_light_filter"] as? Int {
            result += "Blue Light Filter: \(blueLight)%\n"
        }
        
        if let textSharpness = settings["text_sharpness"] as? Int {
            result += "Text Sharpness: \(textSharpness)%\n"
        }
        
        if let responseTime = settings["response_time"] as? Int {
            result += "Response Time: \(responseTime)ms\n"
        }
        
        if let refreshRate = settings["refresh_rate"] as? Int {
            result += "Refresh Rate: \(refreshRate)Hz\n"
        }
        
        if let hdr = settings["hdr"] as? Bool {
            result += "HDR: \(hdr ? "Enabled" : "Disabled")\n"
        }
        
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
