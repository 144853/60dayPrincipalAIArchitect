import React from 'react';
import { Scan, Database, Cpu, GitBranch, LineChart, CheckCircle } from 'lucide-react';

export default function ToothMovementPipeline() {
  const Stage = ({ icon: Icon, title, details, color, number }) => (
    <div className={`relative p-4 rounded-lg border-2 ${color} shadow-lg`}>
      <div className="absolute -top-3 -left-3 w-8 h-8 bg-indigo-600 text-white rounded-full flex items-center justify-center font-bold">
        {number}
      </div>
      <Icon className="w-10 h-10 mb-2 mx-auto text-gray-700" />
      <h3 className="font-bold text-center mb-2 text-sm">{title}</h3>
      <p className="text-xs text-gray-700 leading-tight">{details}</p>
    </div>
  );

  const Arrow = ({ label }) => (
    <div className="flex items-center justify-center my-3">
      <div className="flex-1 h-0.5 bg-indigo-400"></div>
      <div className="px-3 text-xs font-semibold text-indigo-700 bg-indigo-100 rounded-full">
        {label}
      </div>
      <div className="flex-1 h-0.5 bg-indigo-400"></div>
      <div className="w-0 h-0 border-t-4 border-b-4 border-l-8 border-transparent border-l-indigo-400"></div>
    </div>
  );

  return (
    <div className="w-full max-w-6xl mx-auto p-8 bg-gradient-to-br from-blue-50 via-purple-50 to-indigo-50">
      <h1 className="text-3xl font-bold text-center mb-3 text-indigo-900">
        3D Tooth Movement Prediction
      </h1>
      <h2 className="text-lg text-center mb-8 text-indigo-700">
        End-to-End ML Pipeline
      </h2>

      <div className="space-y-2">
        <Stage 
          number="1"
          icon={Scan} 
          title="Data Acquisition" 
          details="iTero scanner captures 3D mesh (STL format) of upper/lower arches. Patient metadata: age, gender, compliance history. Historical treatment records from 10M+ cases."
          color="bg-blue-100 border-blue-400" 
        />

        <Arrow label="Raw Data" />

        <Stage 
          number="2"
          icon={Database} 
          title="Data Preprocessing" 
          details="Convert mesh to voxel grid (128³). Segment individual teeth using U-Net. Normalize coordinates, align to standard reference frame. Extract 200+ features: tooth angles, distances, crowding index, occlusion metrics."
          color="bg-green-100 border-green-400" 
        />

        <Arrow label="Structured Features" />

        <Stage 
          number="3"
          icon={Cpu} 
          title="Feature Extraction (CNN)" 
          details="3D ResNet-50 processes voxelized scan. Outputs 96-dimensional feature vector per tooth capturing shape, position, and surrounding anatomy. Transfer learning from 5M pre-trained cases."
          color="bg-purple-100 border-purple-400" 
        />

        <Arrow label="Spatial Features" />

        <Stage 
          number="4"
          icon={GitBranch} 
          title="Sequential Modeling (RNN)" 
          details="Bidirectional LSTM (3 layers, 512 units) models temporal dependencies. Inputs: CNN features + aligner specifications + treatment timeline. Predicts tooth position at each timestep (typically 30-50 stages)."
          color="bg-orange-100 border-orange-400" 
        />

        <Arrow label="Movement Trajectory" />

        <Stage 
          number="5"
          icon={LineChart} 
          title="Prediction & Optimization" 
          details="Outputs: 3D coordinates per tooth per timestep, rotation matrices, confidence scores. Physics-based constraints ensure biologically feasible movements. Reinforcement learning optimizes aligner sequence to minimize treatment time."
          color="bg-pink-100 border-pink-400" 
        />

        <Arrow label="Treatment Plan" />

        <Stage 
          number="6"
          icon={CheckCircle} 
          title="Clinical Integration" 
          details="Interactive 3D visualization shows predicted outcome. Orthodontist reviews/modifies AI plan. System generates manufacturing specs for each aligner. Monitoring dashboard tracks actual vs predicted during treatment."
          color="bg-teal-100 border-teal-400" 
        />
      </div>

      <div className="mt-8 grid grid-cols-3 gap-4">
        <div className="p-4 bg-white rounded-lg border-2 border-indigo-300 shadow">
          <h3 className="font-bold text-sm mb-2 text-indigo-900">Performance</h3>
          <ul className="text-xs space-y-1">
            <li>• Accuracy: 95%+ prediction</li>
            <li>• Position error: &lt;0.3mm</li>
            <li>• Processing: &lt;3 seconds</li>
            <li>• Success rate: 88% first-time-right</li>
          </ul>
        </div>

        <div className="p-4 bg-white rounded-lg border-2 border-indigo-300 shadow">
          <h3 className="font-bold text-sm mb-2 text-indigo-900">Infrastructure</h3>
          <ul className="text-xs space-y-1">
            <li>• Cloud: AWS/Azure GPU clusters</li>
            <li>• Training: 48 hours on 16x A100</li>
            <li>• Storage: 2PB treatment data</li>
            <li>• API: 99.9% uptime SLA</li>
          </ul>
        </div>

        <div className="p-4 bg-white rounded-lg border-2 border-indigo-300 shadow">
          <h3 className="font-bold text-sm mb-2 text-indigo-900">Business Impact</h3>
          <ul className="text-xs space-y-1">
            <li>• 60% faster planning</li>
            <li>• 40% fewer revisions</li>
            <li>• 25% higher patient satisfaction</li>
            <li>• $200M annual savings</li>
          </ul>
        </div>
      </div>

      <div className="mt-6 p-4 bg-gradient-to-r from-indigo-100 to-purple-100 rounded-lg border-2 border-indigo-400">
        <h3 className="font-bold mb-2 text-indigo-900">Pipeline Summary</h3>
        <p className="text-sm text-gray-800 leading-relaxed">
          The end-to-end pipeline combines computer vision (CNN) for spatial understanding with sequential modeling (RNN) for temporal prediction. 
          Data flows from iTero scanners through preprocessing and feature extraction, then through deep learning models trained on millions of historical cases. 
          The hybrid CNN-RNN architecture captures both the complex 3D anatomy and the temporal dynamics of tooth movement. 
          Physics-informed constraints ensure predictions are biologically feasible, while reinforcement learning optimizes the treatment sequence. 
          The system processes a new case in under 3 seconds, achieving 95%+ accuracy in predicting final tooth positions 12-18 months in advance. 
          Clinical integration allows orthodontists to review, modify, and approve AI-generated plans before manufacturing begins. 
          Continuous learning from treatment outcomes improves the model over time, creating a virtuous cycle of better predictions and outcomes.
        </p>
      </div>
    </div>
  );
}