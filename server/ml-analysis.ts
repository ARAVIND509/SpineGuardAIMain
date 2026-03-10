import type {
  AnalysisResults,
  ClinicalFinding,
  DiagnosticFinding,
  SeverityLevel,
  SoftTissueDegenerationResult,
  PostureSimulationResult,
  HiddenAbnormalityResult,
  BloodFlowAnalysisResult
} from "@shared/schema";

import { exec } from "child_process";

import { analyzeMedicalImageWithML } from "./ml-model";
import { generateGradCAMHeatmaps } from "./gradcam-generator";

import {
  generateFinding,
  generatePostureAnalysis,
  generateSoftTissueAnalysis,
  generateHiddenAbnormalityAnalysis,
  generateBloodFlowAnalysis
} from "./analysis-utils";

export async function analyzeWithMedicalModel(
  imageBuffer: Buffer,
  imageType: string,
  modelType: 'ResNet50' = 'ResNet50'
): Promise<AnalysisResults> {

  console.log(`Analyzing with ${modelType} medical model for ${imageType} scan`);

  const mlResults = await analyzeMedicalImageWithML(imageBuffer, modelType);

  const conditions = mlResults.predictions;

  const regionMap: Record<string, string> = {
    "Disc Herniation": "L4-L5",
    "Scoliosis": "Thoracic",
    "Spinal Stenosis": "L3-L4",
    "Degenerative Disc Disease": "L5-S1",
    "Vertebral Fracture": "Lumbar",
    "Spondylolisthesis": "L4-L5",
    "Infection": "Lumbar",
    "Tumor": "Thoracic"
  };

  const abnormalPredictions = conditions.filter(
    p => p.severity !== 'normal' && p.confidence >= 28
  );

  console.log(`Generating Grad-CAM heatmaps for ${abnormalPredictions.length} abnormal predictions...`);

  const gradCamHeatmaps = abnormalPredictions.length > 0
    ? await generateGradCAMHeatmaps(
        imageBuffer,
        abnormalPredictions.map(p => ({
          ...p,
          confidence: p.confidence
        }))
      )
    : [];

  const discHerniation = conditions.find(c => c.condition === 'Disc Herniation');
  const scoliosis = conditions.find(c => c.condition === 'Scoliosis');
  const spinalStenosis = conditions.find(c => c.condition === 'Spinal Stenosis');
  const degenerativeDisc = conditions.find(c => c.condition === 'Degenerative Disc Disease');
  const vertebralFracture = conditions.find(c => c.condition === 'Vertebral Fracture');
  const spondylolisthesis = conditions.find(c => c.condition === 'Spondylolisthesis');
  const infection = conditions.find(c => c.condition === 'Infection');
  const tumor = conditions.find(c => c.condition === 'Tumor');

  const abnormal = conditions.filter(
    c => c.severity !== 'normal' && c.confidence >= 28
  );

  let primaryFinding: ClinicalFinding | undefined = undefined;

  if (abnormal.length > 0) {

    const severityRank: Record<string, number> = {
      severe: 3,
      moderate: 2,
      mild: 1,
      normal: 0
    };

    const sorted = [...abnormal].sort((a, b) => {

      if (severityRank[a.severity] !== severityRank[b.severity]) {
        return severityRank[b.severity] - severityRank[a.severity];
      }

      return b.confidence - a.confidence;

    });

    const primary = sorted[0];

    primaryFinding = {
      condition: primary.condition,
      severity: primary.severity,
      confidence: primary.confidence,
      location: regionMap[primary.condition] || "Spine"
    };

  }

  const analysisResults: AnalysisResults = {

    discHerniation: generateFinding(discHerniation, "Disc Herniation"),
    scoliosis: generateFinding(scoliosis, "Scoliosis"),
    spinalStenosis: generateFinding(spinalStenosis, "Spinal Stenosis"),
    degenerativeDisc: generateFinding(degenerativeDisc, "Degenerative Disc Disease"),
    vertebralFracture: generateFinding(vertebralFracture, "Vertebral Fracture"),
    spondylolisthesis: generateFinding(spondylolisthesis, "Spondylolisthesis"),
    infection: generateFinding(infection, "Infection"),
    tumor: generateFinding(tumor, "Tumor"),

    postureSimulation: generatePostureAnalysis(conditions, imageType),
    softTissueDegeneration: generateSoftTissueAnalysis(conditions),
    hiddenAbnormality: generateHiddenAbnormalityAnalysis(conditions),
    bloodFlowAnalysis: generateBloodFlowAnalysis(conditions),

    findings: conditions.map(c => ({
      condition: c.condition,
      severity: c.severity,
      confidence: c.confidence,
      location: regionMap[c.condition] || "Spine"
    })),

    primaryFinding: primaryFinding,

    mlPredictions: mlResults,

    gradCamHeatmaps: gradCamHeatmaps.length > 0
      ? gradCamHeatmaps
      : undefined,

    heatmapTargets: gradCamHeatmaps.length > 0
      ? gradCamHeatmaps.flatMap(h =>
          h.affectedRegions.map(r => ({
            condition: h.condition,
            region: r.region,
            intensity: r.intensity,
            severity: h.severity || 'severe'
          }))
        )
      : undefined
  };

  console.log(`Analysis complete with ${gradCamHeatmaps.length} Grad-CAM heatmaps generated`);

  return analysisResults;
}

export function analyzeScan(imagePath: string) {

  return new Promise((resolve, reject) => {

    exec(`python spine_analysis.py ${imagePath}`, (error, stdout, stderr) => {

      if (error) {
        console.error("Python error:", error);
        reject(error);
        return;
      }

      try {

        const result = JSON.parse(stdout);

        resolve(result);

      } catch (err) {

        reject(err);

      }

    });

  });

}