#pragma once
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

struct TrackviewConfig {
    float cruiseSpeed = 0.25f, maxSpeed = 10.0f, minSpeed = 0.05f;
    float curveSpeed = 10.0f, degradedSpeed = 10.0f, acceleration = 0.30f;
    float deceleration = 0.60f, qualityDeceleration = 1.20f;
    float curveStart = 0.20f, qualityRecoverTime = 0.30f;
    int curveLookaheadRow = 10;
    float maxSingleFraction = 0.35f;
    float runP1 = 2.2f, runP2 = 0.007f, turnD = 3.5f;
    float referenceDt = 1.f / 30.f, derivativeFilter = 0.08f;
    float maxErrorRate = 960.f, highSpeedGain = 1.f;
    float lostTimeout = 0.20f, recoverTime = 0.25f, maxFrameGap = 0.25f;
    // widthLifetime is retained for old JSON files: age warning in diagnostics,
    // not expiry of a static-camera width profile. Contradictory measurements revalidate it.
    float widthAlpha = 0.1f, widthLifetime = 15.f;
    int borderMargin = 3, minPoints = 8, minSpan = 30;
    int minWidth = 30, maxWidth = 300, rowCutUp = 10, rowCutBottom = 20;
    int commandTimeoutMs = 250, binary = -1;
    bool diagnostics = true;
    std::string logDirectory = "trackview_logs";

    void validate() const {
        auto positive = [](float x) { return std::isfinite(x) && x > 0; };
        if (!positive(cruiseSpeed) || !positive(maxSpeed) || maxSpeed > 10.f ||
            !positive(minSpeed) || minSpeed > cruiseSpeed || cruiseSpeed > maxSpeed ||
            !positive(curveSpeed) || curveSpeed > maxSpeed ||
            !positive(degradedSpeed) || degradedSpeed > maxSpeed ||
            !positive(maxSingleFraction) || maxSingleFraction >= 1 ||
            !positive(acceleration) || acceleration > 2 || !positive(runP1) || runP1 > 20 ||
            !positive(deceleration) || deceleration > 5 ||
            !positive(qualityDeceleration) || qualityDeceleration > 5 ||
            !std::isfinite(curveStart) || curveStart < 0 || curveStart >= 0.95f ||
            !std::isfinite(qualityRecoverTime) || qualityRecoverTime < 0 || qualityRecoverTime > 5 ||
            curveLookaheadRow < 0 || curveLookaheadRow > 200 ||
            !std::isfinite(runP2) || runP2 < 0 || runP2 > 1 ||
            !std::isfinite(turnD) || turnD < 0 || turnD > 30 ||
            !positive(referenceDt) || referenceDt > 0.1f ||
            !positive(derivativeFilter) || derivativeFilter > 1 ||
            !positive(maxErrorRate) || maxErrorRate > 5000 ||
            !positive(highSpeedGain) || highSpeedGain > 2 ||
            !positive(lostTimeout) || lostTimeout > 1 || !positive(recoverTime) || recoverTime > 5 ||
            !positive(maxFrameGap) || maxFrameGap > 1 ||
            !positive(widthAlpha) || widthAlpha > 1 || !positive(widthLifetime) || widthLifetime > 60 ||
            borderMargin < 1 || borderMargin > 20 || minPoints < 5 || minPoints > 100 ||
            minSpan < 10 || minSpan > 200 || minWidth < 10 ||
            maxWidth <= minWidth || maxWidth >= 320 || rowCutUp < 0 || rowCutUp > 60 ||
            rowCutBottom < 0 || rowCutBottom > 60 || commandTimeoutMs < 100 ||
            commandTimeoutMs > 1000 || binary < -1 || binary > 255 ||
            logDirectory.empty())
            throw std::invalid_argument("Invalid trackview configuration values");
    }
    void load(const std::string &path, bool required = false) {
        std::ifstream file(path);
        if (!file) {
            if (required) throw std::runtime_error("Cannot read config: " + path);
            validate(); return;
        }
        nlohmann::json j; file >> j;
        if (!j.is_object()) throw std::invalid_argument("Config must be a JSON object");
        // Reject spelling mistakes rather than silently ignoring a speed setting.
        const auto known = toJson();
        for (auto it = j.begin(); it != j.end(); ++it)
            if (!known.contains(it.key()) && it.key() != "mixedSpeed")
                throw std::invalid_argument("Unknown config key: " + it.key());
#define TV_READ(name) if (j.contains(#name)) j.at(#name).get_to(name)
        TV_READ(cruiseSpeed); TV_READ(maxSpeed); TV_READ(minSpeed);
        TV_READ(curveSpeed); TV_READ(degradedSpeed); TV_READ(acceleration);
        TV_READ(deceleration); TV_READ(qualityDeceleration); TV_READ(curveStart);
        TV_READ(qualityRecoverTime); TV_READ(curveLookaheadRow);
        TV_READ(maxSingleFraction);
        // Accept v4 configuration files without retaining their extra speed cap.
        if (j.contains("mixedSpeed"))
            std::cerr << "[Config] mixedSpeed ignored; healthy MIXED uses limit.\n";
        TV_READ(runP1); TV_READ(runP2); TV_READ(turnD); TV_READ(referenceDt);
        TV_READ(derivativeFilter); TV_READ(maxErrorRate); TV_READ(highSpeedGain);
        TV_READ(lostTimeout); TV_READ(recoverTime); TV_READ(maxFrameGap);
        TV_READ(widthAlpha); TV_READ(widthLifetime); TV_READ(borderMargin);
        TV_READ(minPoints); TV_READ(minSpan); TV_READ(minWidth); TV_READ(maxWidth);
        TV_READ(rowCutUp); TV_READ(rowCutBottom); TV_READ(commandTimeoutMs);
        TV_READ(binary); TV_READ(diagnostics); TV_READ(logDirectory);
#undef TV_READ
        validate();
    }
    nlohmann::json toJson() const {
        return {{"cruiseSpeed", cruiseSpeed}, {"maxSpeed", maxSpeed},
            {"minSpeed", minSpeed}, {"curveSpeed", curveSpeed},
            {"degradedSpeed", degradedSpeed}, {"acceleration", acceleration},
            {"deceleration", deceleration}, {"qualityDeceleration", qualityDeceleration},
            {"curveStart", curveStart}, {"qualityRecoverTime", qualityRecoverTime},
            {"curveLookaheadRow", curveLookaheadRow},
            {"maxSingleFraction", maxSingleFraction},
            {"runP1", runP1}, {"runP2", runP2}, {"turnD", turnD},
            {"referenceDt", referenceDt}, {"derivativeFilter", derivativeFilter},
            {"maxErrorRate", maxErrorRate}, {"highSpeedGain", highSpeedGain},
            {"lostTimeout", lostTimeout}, {"recoverTime", recoverTime},
            {"maxFrameGap", maxFrameGap}, {"widthAlpha", widthAlpha},
            {"widthLifetime", widthLifetime}, {"borderMargin", borderMargin},
            {"minPoints", minPoints}, {"minSpan", minSpan}, {"minWidth", minWidth},
            {"maxWidth", maxWidth}, {"rowCutUp", rowCutUp},
            {"rowCutBottom", rowCutBottom}, {"commandTimeoutMs", commandTimeoutMs},
            {"binary", binary}, {"diagnostics", diagnostics},
            {"logDirectory", logDirectory}};
    }
};
