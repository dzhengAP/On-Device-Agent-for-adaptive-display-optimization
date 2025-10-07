import SwiftUI
import Combine

/// ViewModel that wires the app to the real experiment system (`LLMBenchmark`)
final class AppViewModel: ObservableObject {

    // MARK: - App state

    @Published var currentApp: String = "None"
    @Published var appType: String = ""
    @Published var recommendations: String = "Waiting for app detection..."
    @Published var parsedSettings: DisplaySettings?
    @Published var autoApplySettings: Bool = false

    @Published var isBenchmarkViewVisible = false
    @Published var isRealChartsViewVisible = false
    @Published var isChatExpanded: Bool = false

    // MARK: - Engines / controllers

    private var llmEngine: LLMEngine?
    private var recommendationEngine: RecommendationEngine?
    private var hybridEngine: HybridSettingsEngine?
    private let displayController = DisplayController()

    // MARK: - Chat

    lazy var chatViewModel: ChatViewModel = {
        let vm = ChatViewModel()
        vm.setAppViewModel(self)
        return vm
    }()

    // MARK: - Lifecycle

    init() {
        setupAppMonitor()
        initializeEngines()
    }

    func getLLMEngine() -> LLMEngine? { llmEngine }

    // MARK: - UI toggles

    func showBenchmarkView() { isBenchmarkViewVisible = true }
    func showRealChartsView() { isRealChartsViewVisible = true }

    // MARK: - App monitoring

    private func setupAppMonitor() {
        AppMonitor.shared.onAppChanged = { [weak self] appName, appType in
            guard let self else { return }
            DispatchQueue.main.async {
                self.currentApp = appName
                self.appType = appType
                self.generateRecommendations(appName: appName, appType: appType)
            }
        }
        AppMonitor.shared.startMonitoring()
    }

    // MARK: - Engine init

    private func initializeEngines() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            self.llmEngine = LLMEngine()
            self.recommendationEngine = RecommendationEngine()
            self.hybridEngine = HybridSettingsEngine()

            DispatchQueue.main.async {
                self.generateRecommendations(
                    appName: self.currentApp,
                    appType: self.appType.isEmpty ? "general purpose app" : self.appType
                )
            }
        }
    }

    // MARK: - Recommendations

    func generateRecommendations(appName: String, appType: String) {
        guard let engine = hybridEngine else {
            recommendations = "Engines not initialized."
            return
        }

        recommendations = "Generating recommendations..."
        let result = engine.getRecommendations(for: appName, appType: appType)

        DispatchQueue.main.async {
            self.recommendations = result
            self.parsedSettings = DisplaySettings.fromLLMOutput(result)

            if self.autoApplySettings, let s = self.parsedSettings {
                self.displayController.applySettings(s)
            }
        }
    }

    func testEngineType(_ engineType: LLMBenchmark.RecommendationEngineType,
                        appName: String,
                        appType: String) -> String {
        switch engineType {
        case .llmOnly:
            guard let engine = recommendationEngine else { return "LLM Engine not available" }
            return engine.getRecommendations(for: appName, appType: appType)
        case .ragOnly:
            return DisplaySettingsDatabase.shared.getFormattedSettings(for: appName, appType: appType)
        case .hybrid:
            guard let engine = hybridEngine else { return "Hybrid Engine not available" }
            return engine.getRecommendations(for: appName, appType: appType)
        }
    }

    func switchToEngine(_ engineType: LLMBenchmark.RecommendationEngineType) {
        let result = testEngineType(engineType, appName: currentApp, appType: appType)
        DispatchQueue.main.async {
            self.recommendations = "[\(engineType.rawValue)]\n\(result)"
            self.parsedSettings = DisplaySettings.fromLLMOutput(result)
        }
    }

    // MARK: - Apply

    func applyCurrentSettings() {
        guard let settings = parsedSettings else { return }
        displayController.applySettings(settings)
    }

    func toggleChat() {
        withAnimation { isChatExpanded.toggle() }
    }

    // MARK: - Real experiment bridges (thin wrappers around LLMBenchmark)

    func runRealAccuracyEvaluation(completion: @escaping ([LLMBenchmark.AccuracyResult]) -> Void) {
        LLMBenchmark.shared.runAccuracyEvaluation(viewModel: self, completion: completion)
    }

    func runRealPerformanceBenchmarking(completion: @escaping ([LLMBenchmark.PerformanceResult]) -> Void) {
        guard let engine = llmEngine else {
            print("LLM Engine not available for performance benchmarking")
            completion([])
            return
        }
        LLMBenchmark.shared.runPerformanceBenchmarking(engine: engine, completion: completion)
    }

    func runRealScalabilityTesting(completion: @escaping ([LLMBenchmark.ScalabilityResult]) -> Void) {
        LLMBenchmark.shared.runScalabilityTesting(viewModel: self, completion: completion)
    }

    func runRealUserExperienceAnalysis(completion: @escaping ([LLMBenchmark.UserExperienceMetric]) -> Void) {
        LLMBenchmark.shared.runUserExperienceAnalysis(completion: completion)
    }

    // MARK: - Export (real data only)

    func exportRealExperimentData(_ experimentData: ExperimentData) {
        let json = LLMBenchmark.shared.exportResults(
            accuracyResults: experimentData.accuracyResults,
            performanceResults: experimentData.performanceResults,
            scalabilityResults: experimentData.scalabilityResults,
            userExperienceMetrics: experimentData.userExperienceMetrics
        )
        saveToFile(content: json, filename: "real_experiment_data.json")
    }

    func exportRealExperimentReport(_ experimentData: ExperimentData) {
        let report = LLMBenchmark.shared.generateExperimentReport(
            accuracyResults: experimentData.accuracyResults,
            performanceResults: experimentData.performanceResults,
            scalabilityResults: experimentData.scalabilityResults,
            userExperienceMetrics: experimentData.userExperienceMetrics
        )
        saveToFile(content: report, filename: "real_experiment_report.md")
    }

    private func saveToFile(content: String, filename: String) {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let url = dir.appendingPathComponent(filename)
        do {
            try content.write(to: url, atomically: true, encoding: .utf8)
            print("📁 Exported: \(url.path)")
        } catch {
            print("❌ Export failed: \(error)")
        }
    }

    // MARK: - Debug

    func getDebugInfo() -> (prompt: String, context: String, output: String, executionTime: Double, memoryUsage: Double)? {
        if let engine = hybridEngine {
            return engine.getDebugInfo()
        } else if let engine = recommendationEngine {
            let (prompt, output) = engine.getDebugInfo()
            return (prompt, "", output, 0.0, 0.0)
        }
        return nil
    }
}

