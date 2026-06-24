# RL_PickAndPlace

Reinforcement Learning Algorithms in a Pick & Place System for Autonomous Robotic Control — a Niryo robotic arm trained with PPO in Unity ML-Agents through a 4-phase curriculum (Reach → Attach → Transport → Lower-and-Release).

The final code is in this folder 👉 `Assets/Scripts/` (`NiryoReachAgent.cs`, `STEP2_Grasp.cs`, `STEP3_Transport.cs`, `Step4_Release.cs`, `Safeplacement.cs`, `SimpleGripperController.cs`)

Unity editor version: `6000.3.9f1`
ML-Agents package version: `4.0.3`

This zip also includes:
- 📄 Full project report — `Report.pdf`
- 🤖 Final trained models for all 4 curriculum phases, in the `Assets/` parent folder — `Step1.onnx`, `Step2.onnx`, `step3.onnx`, `final.onnx`
- 📊 TensorBoard result screenshots — `tensor board images/` (`all.png`, `step1.png`, `step2.png`, `step3.png`, `step4.png`)
