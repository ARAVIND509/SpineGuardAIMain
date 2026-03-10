import type { Express } from "express";
import { createServer, type Server } from "http";
import multer from "multer";
import { storage } from "./storage";
import { analyzeWithMedicalModel } from "./ml-analysis";
import { analyzeWithSCT } from "./sct-bridge";
import { insertPatientSchema, insertScanSchema, insertAnalysisSchema } from "@shared/schema";
import { parseDICOM } from "./dicom-parser";
import { updateAnalysisProgress } from "./websocket-handler";
import { ensureAuthenticated } from "./auth";

// Configure multer for memory storage
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 50 * 1024 * 1024,
  },
});

export async function registerRoutes(app: Express): Promise<Server> {

  // Authentication middleware
  app.use("/api", (req, res, next) => {
    if (req.path === "/register" || req.path === "/login" || req.path === "/logout" || req.path === "/user") {
      return next();
    }
    ensureAuthenticated(req, res, next);
  });

  // ---------------- PATIENT ROUTES ----------------

  app.get("/api/patients", async (req, res) => {
    try {
      const patients = await storage.getAllPatients();
      res.json(patients);
    } catch (error) {
      res.status(500).json({ error: (error as Error).message });
    }
  });

  app.get("/api/patients/:id", async (req, res) => {
    try {
      const patient = await storage.getPatient(req.params.id);
      if (!patient) {
        return res.status(404).json({ error: "Patient not found" });
      }
      res.json(patient);
    } catch (error) {
      res.status(500).json({ error: (error as Error).message });
    }
  });

  app.post("/api/patients", async (req, res) => {
    try {
      const validatedData = insertPatientSchema.parse(req.body);
      const patient = await storage.createPatient(validatedData);
      res.status(201).json(patient);
    } catch (error) {
      res.status(400).json({ error: (error as Error).message });
    }
  });

  // ---------------- SCAN ROUTES ----------------

  app.post("/api/upload", upload.single("image"), async (req, res) => {
    try {

      if (!req.file) {
        return res.status(400).json({ error: "No image file uploaded" });
      }

      const { patientCaseId, imageType } = req.body;

      if (!patientCaseId || !imageType) {
        return res.status(400).json({ error: "Missing required fields" });
      }

      let imageUrl: string;
      let metadata: any = null;

      const isDICOM =
        req.file.mimetype === "application/dicom" ||
        req.file.originalname.toLowerCase().endsWith(".dcm");

      if (isDICOM) {

        const dicomData = await parseDICOM(req.file.buffer);

        imageUrl = `data:image/png;base64,${dicomData.imageBuffer.toString("base64")}`;
        metadata = dicomData.metadata;

      } else {

        const base64Image = req.file.buffer.toString("base64");
        imageUrl = `data:${req.file.mimetype};base64,${base64Image}`;

      }

      const scanData = {
        patientCaseId,
        imageUrl,
        imageType,
        metadata,
      };

      const validatedScanData = insertScanSchema.parse(scanData);

      const scan = await storage.createScan(validatedScanData);

      res.status(201).json({ scan });

    } catch (error) {

      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      res.status(500).json({ error: errorMessage });

    }
  });

  // ---------------- AI ANALYSIS ROUTE ----------------

  app.post("/api/analyze/:scanId", async (req, res) => {

    try {

      const scanId = req.params.scanId;

      const { modelType = "ResNet50" } = req.body;

      const existingAnalysis = await storage.getAnalysis(scanId);

      if (existingAnalysis) {
        return res.json({ analysis: existingAnalysis });
      }

      const scan = await storage.getScan(scanId);

      if (!scan) {
        return res.status(404).json({ error: "Scan not found" });
      }

      let base64Image = scan.imageUrl;

      if (scan.imageUrl.includes(",")) {
        base64Image = scan.imageUrl.split(",")[1];
      }

      const imageBuffer = Buffer.from(base64Image, "base64");

      const startTime = Date.now();

      updateAnalysisProgress(scanId, 10, "Preprocessing medical image");

      let analysisResults;

      try {

        // Primary segmentation with Spinal Cord Toolbox
        updateAnalysisProgress(scanId, 30, "Running spinal cord segmentation");

        analysisResults = await analyzeWithSCT(imageBuffer, scan.imageType);

        updateAnalysisProgress(scanId, 80, "Segmentation completed");

      } catch (sctError) {

        // Fallback to pretrained deep learning model
        updateAnalysisProgress(scanId, 40, "Fallback to deep learning model");

        analysisResults = await analyzeWithMedicalModel(
          imageBuffer,
          scan.imageType,
          "ResNet50"
        );

      }

      updateAnalysisProgress(scanId, 90, "Finalizing diagnosis");

      const duration = Date.now() - startTime;

      if (!analysisResults.mlPredictions) {
        analysisResults.mlPredictions = {
          predictions: [],
          modelUsed: "Combined Analysis",
          processingTime: duration,
        };
      } else {
        analysisResults.mlPredictions.processingTime = duration;
      }

      const analysisData = {
        scanId: scan.id,
        results: analysisResults,
      };

      const validatedAnalysisData = insertAnalysisSchema.parse(analysisData);

      const analysis = await storage.createAnalysis(validatedAnalysisData);

      updateAnalysisProgress(scanId, 100, "Analysis complete");

      res.status(201).json({ analysis });

    } catch (error) {

      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      res.status(500).json({ error: errorMessage });

    }

  });

  // ---------------- ANALYSIS FETCH ----------------

  app.get("/api/analysis/:scanId", async (req, res) => {
    try {
      const analysis = await storage.getAnalysis(req.params.scanId);
      if (!analysis) {
        return res.status(404).json({ error: "Analysis not found" });
      }
      res.json(analysis);
    } catch (error) {
      res.status(500).json({ error: (error as Error).message });
    }
  });

  const httpServer = createServer(app);

  return httpServer;
}