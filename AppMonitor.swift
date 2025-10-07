//
//  AppMonitor.swift
//  On Display Computing
//
//  Created by David Zheng on 6/29/25.
//


import AppKit

/// Monitors and detects which application is currently active in the foreground
class AppMonitor {
    /// Singleton instance for app-wide access
    static let shared = AppMonitor()
    
    /// Timer used to check the active application at regular intervals
    private var timer: Timer?
    
    /// Keeps track of the currently active application to detect changes
    private var currentApp: NSRunningApplication?
    
    /// Callback function that is triggered when the active application changes
    /// - Parameters:
    ///   - String: Name of the new active application
    ///   - String: Type/category of the new active application
    var onAppChanged: ((String, String) -> Void)?
    
    /// Begins the process of monitoring which application is active
    func startMonitoring() {
        // Create a timer that checks the active app every second
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.checkActiveApp()
        }
    }
    
    /// Checks which application is currently in the foreground
    private func checkActiveApp() {
        // Get the app currently at the front from macOS workspace
        guard let frontmostApp = NSWorkspace.shared.frontmostApplication else { return }
        
        // Only process if the app has changed since our last check
        if currentApp != frontmostApp {
            // Update our stored reference to the current app
            currentApp = frontmostApp
            
            // Extract useful information about the app
            let appName = frontmostApp.localizedName ?? "Unknown App"
            let bundleID = frontmostApp.bundleIdentifier ?? "unknown"
            
            // Determine what type of app this is based on its bundle ID
            let appType = getAppType(bundleID: bundleID)
            
            // Notify any listeners (like our view model) about the app change
            onAppChanged?(appName, appType)
        }
    }
    
    /// Categorizes an application based on its bundle identifier
    /// - Parameter bundleID: The bundle identifier of the app (e.g. "com.apple.Safari")
    /// - Returns: A string describing the app's category
    private func getAppType(bundleID: String) -> String {
        // Check bundle ID against known application types
        if bundleID.contains("photoshop") || bundleID.contains("lightroom") {
            return "photo editing app"  // Adobe Photoshop, Lightroom
        } else if bundleID.contains("final") || bundleID.contains("premiere") {
            return "video editing app"  // Final Cut Pro, Adobe Premiere
        } else if bundleID.contains("safari") || bundleID.contains("chrome") || bundleID.contains("firefox") {
            return "web browser"  // Web browsers
        } else if bundleID.contains("word") || bundleID.contains("pages") || bundleID.contains("text")||bundleID.contains("excel") {
            return "text editing app"  // Word processors
        } else if bundleID.contains("terminal") || bundleID.contains("iterm") {
            return "terminal app"  // Command-line interfaces
        } else if bundleID.contains("xcode") || bundleID.contains("vscode") {
            return "code editor"  // Programming environments
        } else if bundleID.contains("game") || bundleID.contains("steam") {
            return "gaming app"  // Games or game platforms
        }
        
        // Default category if no specific match is found
        return "general purpose app"
    }
}
