//
//  OnDisplayComputingApp.swift
//  On Display Computing
//
//  Created by David Zheng on 6/29/25.
//


import SwiftUI

/// Main app class
@main
struct OnDisplayComputingApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
            .onAppear {
                    // Initialize and run first test when app appears
                    print("🟢 App launched")
                    // Uncomment this for testing:
                    // LLMtest()
                }
        }
    }
}
