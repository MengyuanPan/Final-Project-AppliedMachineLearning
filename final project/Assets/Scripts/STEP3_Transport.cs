using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Reflection;

// STEP3_Transport.cs
// This script keeps the Step2 agent's Approach behaviour (closing in on the object, closing the gripper, and attaching) and extends it into the Grasp and Transport stage: once attached, the active goal switches from the object itself to a point above the safe placement zone, and the arm is trained to carry the object there while keeping the grasp stable.
// Placement itself - accurately releasing the object at the target - is left to Step4; this stage only needs to reach and hold the placement zone while still grasping.
public class STEP3_Transport : Agent
{
    [Header("Robot Joints — ArticulationBody")]
    public ArticulationBody jointA1;
    public ArticulationBody jointA2;
    public ArticulationBody jointA3;
    public ArticulationBody jointA4;
    public ArticulationBody jointA5;
    public ArticulationBody jointA6;

    [Header("Joint angle limits")]
    public float jointMinAngle = -120f;
    public float jointMaxAngle = 120f;

    [Header("End Point")]
    public Transform graspPoint;

    [Header("Scene")]
    public Transform targetObject;
    public Transform targetPlacement;
    public Safeplacement safePlacementZone;
    public Transform robotBase;

    [Header("Gripper")]
    public MonoBehaviour gripperController;
    public ArticulationBody leftFingerJoint;
    public ArticulationBody rightFingerJoint;

    public float leftFingerOpenTarget = 0.01f;
    public float leftFingerClosedTarget = -0.01f;
    public float rightFingerOpenTarget = -0.01f;
    public float rightFingerClosedTarget = 0.01f;
    [Range(0.01f, 1f)] public float gripperCloseSpeed = 0.08f;

    [Header("Spawn Area — same as Step2")]
    public Vector3 spawnCenter = new Vector3(0.216f, 0.7f, -0.157f);
    public float spawnHalfSize = 0.1f;
    public float forbiddenRadius = 0.15f;
    public int spawnMaxTries = 20;

    [Header("Movement")]
    public float jointSpeed = 0.72f;
    [Range(0.01f, 1f)] public float actionSmoothing = 0.16f;
    [Range(0.05f, 1f)] public float maxAbsAction = 0.42f;

    [Header("Anti-collapse reset / warmup")]
    public int warmupDecisionSteps = 3;
    public bool hardResetJointPositions = true;
    public bool zeroJointVelocityOnReset = true;

    public bool strengthenJointDrives = false;
    public float minDriveStiffness = 10000f;
    public float minDriveDamping = 1000f;
    public float minDriveForceLimit = 1000f;

    [Header("Episode")]
    public int maxEpisodeSteps = 5200;

    [Header("Step2 kept: approach / grasp / attach")]
    public float closeStartDistance = 0.095f;
    public float attachDistance = 0.06f;
    public int attachConfirmSteps = 2;
    public Vector3 attachedLocalPosition = Vector3.zero;
    public Vector3 attachedLocalEuler = Vector3.zero;

    [Header("Step3: transport")]
    public Vector3 placementGoalOffset = new Vector3(0f, 0.12f, 0f);
    public float placementRadius = 0.075f;
    public int placementHoldSteps = 35;

    [Header("Carry height")]
    public float minCarryHeightAboveSpawn = 0.08f;
    public float desiredCarryHeightAboveSpawn = 0.14f;
    public float maxCarryHeightAboveSpawn = 0.34f;
    public int lowCarryStepLimit = 120;

    [Header("Physics simplification")]
    public bool disableTargetCollidersDuringStep3 = true;

    [Header("Safety limits")]
    public float minAllowedGraspPointY = 0.35f;
    public float maxWorkspaceDistanceFromSpawnCenter = 1.20f;
    public float maxAverageJointVelocity = 12f;
    public int unsafeVelocityStepLimit = 4;

    [Header("Reward — approach / attach")]
    public float approachImprovementScale = 5f;
    public float approachDistancePenaltyScale = 0.02f;
    public float closeZoneReward = 0.004f;
    public float attachReadyReward = 0.015f;
    public float attachBonus = 1.0f;

    [Header("Reward — transport")]
    public float transportImprovementScale = 6.5f;
    public float transportDistancePenaltyScale = 0.035f;
    public float carryHeightReward = 0.010f;
    public float lowCarryPenalty = -0.012f;
    public float placeZoneReward = 0.030f;
    public float lockReward = 0.006f;
    public float lockErrorPenaltyScale = 0.040f;

    [Header("Common reward / penalty")]
    public float timePenalty = -0.001f;
    public float stuckPenalty = -0.02f;
    public float successReward = 4.0f;
    public float timeoutPenalty = -1.0f;
    public float unsafePenalty = -1.5f;
    public float jointVelocityPenaltyScale = 0.002f;
    public float actionChangePenaltyScale = 0.001f;

    [Header("Stuck detection")]
    public float minImprovement = 0.001f;
    public int stuckStepLimit = 240;

    [Header("Debug")]
    public bool drawDebugGizmos = true;

    private float _restA1;
    private float _restA2;
    private float _restA3;
    private float _restA4;
    private float _restA5;
    private float _restA6;

    private float _prevReachDist;
    private float _bestReachDist;
    private float _prevTransportDist;
    private float _bestTransportDist;

    private int _stuckSteps;
    private int _stepCount;
    private int _unsafeVelocitySteps;
    private int _lowCarrySteps;
    private int _placementHoldCounter;

    private Rigidbody _targetRb;
    private Transform _targetOriginalParent;
    private Quaternion _targetOriginalRotation;

    private bool _attached;
    private int _attachZoneSteps;
    private int _attachedSteps;
    private float _gripperClose01;

    private readonly float[] _smoothedActions = new float[6];
    private readonly float[] _lastAppliedActions = new float[6];

    private const float OBS_NORM = 0.5f;

    // One-time setup when the agent is created (not per-episode): caches each joint's resting angle, stores the references needed to reset the target object later, and strips the target's physics so it can be driven purely by script once the Grasp and Transport stage begins.
    public override void Initialize()
    {
        if (strengthenJointDrives)
        {
            StrengthenAllDrives();
        }

        _restA1 = GetCurrentAngleOrTarget(jointA1);
        _restA2 = GetCurrentAngleOrTarget(jointA2);
        _restA3 = GetCurrentAngleOrTarget(jointA3);
        _restA4 = GetCurrentAngleOrTarget(jointA4);
        _restA5 = GetCurrentAngleOrTarget(jointA5);
        _restA6 = GetCurrentAngleOrTarget(jointA6);

        if (targetObject != null)
        {
            _targetRb = targetObject.GetComponent<Rigidbody>();
            _targetOriginalParent = targetObject.parent;
            _targetOriginalRotation = targetObject.rotation;
        }

        if (targetPlacement == null && safePlacementZone != null)
        {
            targetPlacement = safePlacementZone.transform;
        }

        PrepareTargetPhysicsBeforeAttach();
    }

    // Resets everything needed for a new episode: joints return to their rest pose, the gripper opens, the target object is re-spawned with randomisation, the placement zone is randomised, and both the Approach and Grasp-and-Transport distance trackers are re-initialised.
    // This sets up the initial state for the Perception and State Representation stage before the agent starts acting.
    public override void OnEpisodeBegin()
    {
        ResetJointToSafePose(jointA1, _restA1);
        ResetJointToSafePose(jointA2, _restA2);
        ResetJointToSafePose(jointA3, _restA3);
        ResetJointToSafePose(jointA4, _restA4);
        ResetJointToSafePose(jointA5, _restA5);
        ResetJointToSafePose(jointA6, _restA6);

        for (int i = 0; i < 6; i++)
        {
            _smoothedActions[i] = 0f;
            _lastAppliedActions[i] = 0f;
        }

        _attached = false;
        _attachZoneSteps = 0;
        _attachedSteps = 0;
        _placementHoldCounter = 0;
        _lowCarrySteps = 0;
        _unsafeVelocitySteps = 0;
        _gripperClose01 = 0f;

        OpenGripperImmediately();
        ResetTargetObject();
        RandomizePlacementZone();

        _stepCount = 0;
        _stuckSteps = 0;

        _prevReachDist = GraspToTargetDist();
        _bestReachDist = _prevReachDist;

        _prevTransportDist = TransportGoalDist();
        _bestTransportDist = _prevTransportDist;
    }

    // Observation space for the Perception and State Representation stage: 6 normalised joint angles, the grasp point's position relative to the spawn centre, the position of the currently active goal (the object while approaching, the placement point while carrying), the vector and distance from the gripper to that goal, height above the spawn plane, normalised episode progress, gripper closure / attachment state, and the best distance achieved so far on the current sub-task.
    // The vector length is fixed at 20 dimensions so this Step3 policy can be initialised from the Step2-trained model.
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(NormAngle(jointA1));
        sensor.AddObservation(NormAngle(jointA2));
        sensor.AddObservation(NormAngle(jointA3));
        sensor.AddObservation(NormAngle(jointA4));
        sensor.AddObservation(NormAngle(jointA5));
        sensor.AddObservation(NormAngle(jointA6));

        Vector3 gpRel = graspPoint != null
            ? (graspPoint.position - spawnCenter) / OBS_NORM
            : Vector3.zero;
        sensor.AddObservation(ClampVec(gpRel));

        Vector3 activeGoal = GetActiveGoalPosition();
        Vector3 activeGoalRel = (activeGoal - spawnCenter) / OBS_NORM;
        sensor.AddObservation(ClampVec(activeGoalRel));

        Vector3 toActiveGoal = graspPoint != null
            ? activeGoal - graspPoint.position
            : Vector3.zero;
        sensor.AddObservation(ClampVec(toActiveGoal));

        sensor.AddObservation(Mathf.Clamp(toActiveGoal.magnitude, 0f, 5f));
        sensor.AddObservation(Mathf.Clamp(graspPoint != null ? graspPoint.position.y - spawnCenter.y : 0f, -5f, 5f));
        sensor.AddObservation((float)_stepCount / Mathf.Max(1, maxEpisodeSteps));
        sensor.AddObservation(_attached ? 1f : _gripperClose01);
        sensor.AddObservation(Mathf.Clamp01(_attached ? _bestTransportDist : _bestReachDist));
    }

    // Action space and main control loop: continuous actions are 6 joint deltas, applied in ApplySafeJointActions.
    // A short warmup period holds the rest pose so observations stabilise before the agent starts acting.
    // After warmup, the agent pursues the Approach objective (including the grasp trigger) while not yet attached, or the Grasp and Transport objective once attached, after which smoothness/safety penalties are applied and episode-ending conditions (unsafe state / timeout) are checked.
    public override void OnActionReceived(ActionBuffers actions)
    {
        _stepCount++;

        if (_stepCount <= warmupDecisionSteps)
        {
            HoldRestPoseDuringWarmup();
            UpdateScriptedGripper();
            AddReward(timePenalty);
            _prevReachDist = GraspToTargetDist();
            _prevTransportDist = TransportGoalDist();
            return;
        }

        ApplySafeJointActions(actions);
        UpdateScriptedGripper();

        if (!_attached)
        {
            StepApproachAndAttach();
        }
        else
        {
            StepTransportToPlacement();
        }

        AddSmoothnessAndVelocityPenalty(actions);

        if (IsUnsafeState())
        {
            AddReward(unsafePenalty);
            EndEpisode();
            return;
        }

        if (_stepCount >= maxEpisodeSteps)
        {
            AddReward(timeoutPenalty);
            EndEpisode();
        }
    }

    // Manual keyboard control for testing the joints without a trained policy: each joint has an "increase" key (number row) and a "decrease" key (the QWERTY key beneath it).
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var ca = actionsOut.ContinuousActions;

        for (int i = 0; i < 6; i++)
        {
            ca[i] = 0f;
        }

        if (Input.GetKey(KeyCode.Alpha1)) ca[0] = 1f;
        if (Input.GetKey(KeyCode.Q)) ca[0] = -1f;

        if (Input.GetKey(KeyCode.Alpha2)) ca[1] = 1f;
        if (Input.GetKey(KeyCode.W)) ca[1] = -1f;

        if (Input.GetKey(KeyCode.Alpha3)) ca[2] = 1f;
        if (Input.GetKey(KeyCode.E)) ca[2] = -1f;

        if (Input.GetKey(KeyCode.Alpha4)) ca[3] = 1f;
        if (Input.GetKey(KeyCode.R)) ca[3] = -1f;

        if (Input.GetKey(KeyCode.Alpha5)) ca[4] = 1f;
        if (Input.GetKey(KeyCode.T)) ca[4] = -1f;

        if (Input.GetKey(KeyCode.Alpha6)) ca[5] = 1f;
        if (Input.GetKey(KeyCode.Y)) ca[5] = -1f;
    }

    // Approach stage reward shaping, with grasp trigger: rewards reducing the gripper-to-object distance and staying within the "close" and "attach" zones.
    // The grasp itself - the start of the Grasp and Transport stage - is triggered (and rewarded) once the gripper has remained close enough for long enough while the scripted gripper has closed past a threshold, simulating a successful grasp.
    private void StepApproachAndAttach()
    {
        float currReachDist = GraspToTargetDist();

        AddReward((_prevReachDist - currReachDist) * approachImprovementScale);
        AddReward(-currReachDist * approachDistancePenaltyScale * Time.fixedDeltaTime);
        AddReward(timePenalty);

        if (currReachDist < closeStartDistance)
        {
            AddReward(closeZoneReward);
        }

        if (currReachDist < attachDistance)
        {
            _attachZoneSteps++;
            AddReward(attachReadyReward);
        }
        else
        {
            _attachZoneSteps = 0;
        }

        if (_attachZoneSteps >= attachConfirmSteps && _gripperClose01 >= 0.55f)
        {
            AttachTargetToGraspPoint();
            _prevTransportDist = TransportGoalDist();
            _bestTransportDist = _prevTransportDist;
            _placementHoldCounter = 0;
            _lowCarrySteps = 0;
            AddReward(attachBonus);
        }

        AddStuckPenalty(_prevReachDist, currReachDist);

        if (currReachDist < _bestReachDist)
        {
            _bestReachDist = currReachDist;
        }

        _prevReachDist = currReachDist;
    }

    // Grasp and Transport stage reward shaping: keeps the object kinematically locked to the gripper (maintaining the grasp) while rewarding staying locked with low positional error, lifting to a safe carry height (penalising carrying too low or too high), and reducing distance to the placement goal.
    // Progress reward is scaled by lift height (gatedTransportScale) so the agent prioritises lifting before moving horizontally.
    // Holding the placement zone for enough consecutive steps ends the episode with a success reward - the object is not released here, since Placement/release is handled in Step4.
    private void StepTransportToPlacement()
    {
        KeepTargetLockedToGraspPoint();
        _attachedSteps++;

        float currTransportDist = TransportGoalDist();
        float heightAboveSpawn = CarryHeightAboveSpawn();

        float lockError = targetObject != null && graspPoint != null
            ? Vector3.Distance(targetObject.position, graspPoint.position)
            : 1f;

        AddReward(lockReward);
        AddReward(-lockError * lockErrorPenaltyScale);
        AddReward(timePenalty);

        float lift01 = Mathf.InverseLerp(
            minCarryHeightAboveSpawn,
            desiredCarryHeightAboveSpawn,
            heightAboveSpawn
        );
        lift01 = Mathf.Clamp01(lift01);

        if (heightAboveSpawn >= minCarryHeightAboveSpawn)
        {
            AddReward(carryHeightReward * lift01);
        }
        else
        {
            _lowCarrySteps++;
            AddReward(lowCarryPenalty);
        }

        if (heightAboveSpawn > maxCarryHeightAboveSpawn)
        {
            AddReward(-0.010f);
        }

        float gatedTransportScale = Mathf.Lerp(0.30f, 1.00f, lift01);

        AddReward((_prevTransportDist - currTransportDist) * transportImprovementScale * gatedTransportScale);
        AddReward(-currTransportDist * transportDistancePenaltyScale * Time.fixedDeltaTime);

        if (currTransportDist < _bestTransportDist)
        {
            _bestTransportDist = currTransportDist;
        }

        if (currTransportDist <= placementRadius && heightAboveSpawn >= minCarryHeightAboveSpawn)
        {
            _placementHoldCounter++;
            AddReward(placeZoneReward);

            if (_placementHoldCounter >= placementHoldSteps)
            {
                AddReward(successReward);
                EndEpisode();
                return;
            }
        }
        else
        {
            _placementHoldCounter = 0;
        }

        if (_lowCarrySteps > lowCarryStepLimit)
        {
            AddReward(-0.4f);
            _lowCarrySteps = 0;
        }

        AddStuckPenalty(_prevTransportDist, currTransportDist);
        _prevTransportDist = currTransportDist;
    }

    // Converts the policy's 6 continuous actions into incremental joint-drive targets (the action space implementation).
    // Action magnitude is scaled down ("nearScale") when close to the object during the Approach stage, or while carrying near the placement zone during Grasp and Transport, to encourage finer control and reduce overshoot.
    private void ApplySafeJointActions(ActionBuffers actions)
    {
        float nearScale = 1f;

        if (!_attached)
        {
            float dist = GraspToTargetDist();

            if (dist < closeStartDistance)
            {
                nearScale = 0.85f;
            }

            if (dist < attachDistance * 1.35f)
            {
                nearScale = 0.65f;
            }
        }
        else
        {
            float heightAbove = CarryHeightAboveSpawn();

            nearScale = heightAbove < minCarryHeightAboveSpawn ? 0.78f : 0.68f;

            if (TransportGoalDist() < placementRadius * 1.8f)
            {
                nearScale *= 0.70f;
            }
        }

        AddDeltaTarget(jointA1, SmoothAction(actions.ContinuousActions[0], 0) * jointSpeed * 1.00f * nearScale);
        AddDeltaTarget(jointA2, SmoothAction(actions.ContinuousActions[1], 1) * jointSpeed * 0.95f * nearScale);
        AddDeltaTarget(jointA3, SmoothAction(actions.ContinuousActions[2], 2) * jointSpeed * 0.90f * nearScale);
        AddDeltaTarget(jointA4, SmoothAction(actions.ContinuousActions[3], 3) * jointSpeed * 0.70f * nearScale);
        AddDeltaTarget(jointA5, SmoothAction(actions.ContinuousActions[4], 4) * jointSpeed * 0.60f * nearScale);
        AddDeltaTarget(jointA6, SmoothAction(actions.ContinuousActions[5], 5) * jointSpeed * 0.50f * nearScale);
    }

    // Clamps a raw action to the allowed range and low-pass filters it against the previous smoothed value, reducing jitter in the commanded joint motion.
    private float SmoothAction(float rawAction, int index)
    {
        float clamped = Mathf.Clamp(rawAction, -maxAbsAction, maxAbsAction);
        _smoothedActions[index] = Mathf.Lerp(_smoothedActions[index], clamped, actionSmoothing);
        return _smoothedActions[index];
    }

    // Gripper open/close is scripted rather than learned: it closes once the object is within the "close" zone (or always while attached), at a speed that differs between the Approach and Grasp-and-Transport stages.
    // Drives both the physical finger joints and, optionally, an external gripper controller.
    private void UpdateScriptedGripper()
    {
        float dist = GraspToTargetDist();
        float targetClose = dist < closeStartDistance || _attached ? 1f : 0f;

        float speed = _attached ? gripperCloseSpeed : gripperCloseSpeed * 0.45f;

        _gripperClose01 = Mathf.MoveTowards(
            _gripperClose01,
            targetClose,
            speed * Time.fixedDeltaTime * 50f
        );

        float leftTarget = Mathf.Lerp(leftFingerOpenTarget, leftFingerClosedTarget, _gripperClose01);
        float rightTarget = Mathf.Lerp(rightFingerOpenTarget, rightFingerClosedTarget, _gripperClose01);

        SetRawTarget(leftFingerJoint, leftTarget);
        SetRawTarget(rightFingerJoint, rightTarget);

        DriveOptionalGripperController(_gripperClose01);
    }

    // Forces the gripper fully open at the start of an episode, bypassing the gradual MoveTowards used during normal operation.
    private void OpenGripperImmediately()
    {
        _gripperClose01 = 0f;
        SetRawTarget(leftFingerJoint, leftFingerOpenTarget);
        SetRawTarget(rightFingerJoint, rightFingerOpenTarget);
        DriveOptionalGripperController(0f);
    }

    // Optional hook for an external gripper controller component, called through reflection so this script has no hard dependency on its type.
    // Tries a few common method-name conventions for setting close amount and for opening/closing.
    private void DriveOptionalGripperController(float close01)
    {
        if (gripperController == null)
        {
            return;
        }

        CallOptionalFloatMethod(gripperController, "SetCloseAmount", close01);
        CallOptionalFloatMethod(gripperController, "SetGripperClose", close01);
        CallOptionalFloatMethod(gripperController, "SetGrip", close01);

        if (close01 > 0.5f)
        {
            CallOptionalVoidMethod(gripperController, "CloseGripper");
            CallOptionalVoidMethod(gripperController, "Close");
        }
        else
        {
            CallOptionalVoidMethod(gripperController, "OpenGripper");
            CallOptionalVoidMethod(gripperController, "Open");
        }
    }

    // Reflection helper: invokes a parameterless method on the optional gripper controller if one with this name exists, otherwise does nothing.
    private void CallOptionalVoidMethod(MonoBehaviour target, string methodName)
    {
        MethodInfo m = target.GetType().GetMethod(
            methodName,
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
            null,
            System.Type.EmptyTypes,
            null
        );

        if (m != null)
        {
            m.Invoke(target, null);
        }
    }

    // Reflection helper: invokes a single-float-parameter method on the optional gripper controller if one with this name exists, otherwise does nothing.
    private void CallOptionalFloatMethod(MonoBehaviour target, string methodName, float value)
    {
        MethodInfo m = target.GetType().GetMethod(
            methodName,
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
            null,
            new System.Type[] { typeof(float) },
            null
        );

        if (m != null)
        {
            m.Invoke(target, new object[] { value });
        }
    }

    // Re-parents the target object and re-spawns it at a randomised position within the spawn area (domain randomisation, supporting the generalisation testing described in the brief), then re-applies the simplified Step3 physics setup.
    private void ResetTargetObject()
    {
        if (targetObject == null)
        {
            return;
        }

        targetObject.SetParent(_targetOriginalParent, true);
        targetObject.position = GetValidSpawnPosition();
        targetObject.rotation = _targetOriginalRotation;

        PrepareTargetPhysicsBeforeAttach();
    }

    // Simplifies the target object's physics ahead of grasping: disables gravity and collision detection and makes it kinematic, optionally also disabling its colliders, so its motion is fully script-driven once attached rather than physically simulated.
    private void PrepareTargetPhysicsBeforeAttach()
    {
        if (_targetRb == null && targetObject != null)
        {
            _targetRb = targetObject.GetComponent<Rigidbody>();
        }

        if (_targetRb != null)
        {
            if (!_targetRb.isKinematic)
            {
                ZeroTargetRigidbodyVelocity();
            }

            _targetRb.useGravity = false;
            _targetRb.detectCollisions = false;
            _targetRb.isKinematic = true;
        }

        if (disableTargetCollidersDuringStep3 && targetObject != null)
        {
            Collider[] cols = targetObject.GetComponentsInChildren<Collider>();

            foreach (Collider c in cols)
            {
                c.enabled = false;
            }
        }
    }

    // Implements the grasp at the start of the Grasp and Transport stage: instead of simulating a physical grip, the target object is kinematically parented to the gripper's grasp point at a fixed local offset, guaranteeing a stable hold while it is carried.
    private void AttachTargetToGraspPoint()
    {
        if (targetObject == null || graspPoint == null)
        {
            return;
        }

        _attached = true;
        _attachedSteps = 0;
        _placementHoldCounter = 0;
        _lowCarrySteps = 0;

        if (_targetRb != null)
        {
            if (!_targetRb.isKinematic)
            {
                ZeroTargetRigidbodyVelocity();
            }

            _targetRb.useGravity = false;
            _targetRb.detectCollisions = false;
            _targetRb.isKinematic = true;
        }

        if (disableTargetCollidersDuringStep3)
        {
            Collider[] cols = targetObject.GetComponentsInChildren<Collider>();

            foreach (Collider c in cols)
            {
                c.enabled = false;
            }
        }

        targetObject.SetParent(graspPoint, false);
        targetObject.localPosition = attachedLocalPosition;
        targetObject.localRotation = Quaternion.Euler(attachedLocalEuler);
    }

    // Called every physics step while attached, re-pinning the object's local pose to the grasp point in case parenting or transform state drifts.
    private void KeepTargetLockedToGraspPoint()
    {
        if (targetObject == null || graspPoint == null)
        {
            return;
        }

        if (targetObject.parent != graspPoint)
        {
            targetObject.SetParent(graspPoint, false);
        }

        targetObject.localPosition = attachedLocalPosition;
        targetObject.localRotation = Quaternion.Euler(attachedLocalEuler);

        if (_targetRb != null)
        {
            _targetRb.useGravity = false;
            _targetRb.detectCollisions = false;
            _targetRb.isKinematic = true;
        }
    }

    private void ZeroTargetRigidbodyVelocity()
    {
        if (_targetRb == null)
        {
            return;
        }

        // Unity 6 does not allow setting velocity on a Rigidbody while isKinematic = true, so velocity is only cleared while the body is still non-kinematic.
        // This avoids repeated warnings being logged to the Console once the object becomes kinematic.
        if (_targetRb.isKinematic)
        {
            return;
        }

#if UNITY_6000_0_OR_NEWER
        _targetRb.linearVelocity = Vector3.zero;
#else
        _targetRb.velocity = Vector3.zero;
#endif

        _targetRb.angularVelocity = Vector3.zero;
    }

    // Reward shaping: tracks consecutive steps without meaningful progress on the relevant distance metric, applying a small penalty once the agent has been stuck for too long, to discourage freezing or oscillating in place.
    private void AddStuckPenalty(float prevDist, float currDist)
    {
        if (prevDist - currDist > minImprovement)
        {
            _stuckSteps = 0;
        }
        else
        {
            _stuckSteps++;

            if (_stuckSteps > stuckStepLimit)
            {
                AddReward(stuckPenalty);
            }
        }
    }

    // Regularises the action space: penalises large frame-to-frame action changes (encouraging smooth motion) and penalises high average joint velocity, capped to avoid runaway penalties.
    private void AddSmoothnessAndVelocityPenalty(ActionBuffers actions)
    {
        float change = 0f;

        for (int i = 0; i < 6; i++)
        {
            float a = Mathf.Clamp(actions.ContinuousActions[i], -1f, 1f);
            change += Mathf.Abs(a - _lastAppliedActions[i]);
            _lastAppliedActions[i] = a;
        }

        AddReward(-change * actionChangePenaltyScale * Time.fixedDeltaTime);

        float avgVel = AverageJointVelocityAbs();
        AddReward(-Mathf.Min(avgVel, maxAverageJointVelocity) * jointVelocityPenaltyScale * Time.fixedDeltaTime);
    }

    // Safety / early-termination check: flags NaN positions, the gripper dropping below a minimum height, the arm moving too far from the workspace centre, the carried object falling below the spawn height, or sustained excessive joint velocity.
    private bool IsUnsafeState()
    {
        if (graspPoint == null)
        {
            return false;
        }

        Vector3 p = graspPoint.position;

        if (float.IsNaN(p.x) || float.IsNaN(p.y) || float.IsNaN(p.z))
        {
            return true;
        }

        if (p.y < minAllowedGraspPointY)
        {
            return true;
        }

        if (Vector3.Distance(p, spawnCenter) > maxWorkspaceDistanceFromSpawnCenter)
        {
            return true;
        }

        if (_attached && CarryHeightAboveSpawn() < -0.04f)
        {
            return true;
        }

        if (AverageJointVelocityAbs() > maxAverageJointVelocity)
        {
            _unsafeVelocitySteps++;
        }
        else
        {
            _unsafeVelocitySteps = 0;
        }

        return _unsafeVelocitySteps >= unsafeVelocityStepLimit;
    }

    // Average absolute joint velocity across all six joints, used by both the safety check and the velocity penalty.
    private float AverageJointVelocityAbs()
    {
        float sum = 0f;
        int count = 0;

        AddJointVelocity(jointA1, ref sum, ref count);
        AddJointVelocity(jointA2, ref sum, ref count);
        AddJointVelocity(jointA3, ref sum, ref count);
        AddJointVelocity(jointA4, ref sum, ref count);
        AddJointVelocity(jointA5, ref sum, ref count);
        AddJointVelocity(jointA6, ref sum, ref count);

        return count > 0 ? sum / count : 0f;
    }

    // Accumulates one joint's absolute velocity into a running sum/count, skipping joints that have no velocity-capable degree of freedom.
    private void AddJointVelocity(ArticulationBody ab, ref float sum, ref int count)
    {
        if (ab == null || ab.jointVelocity.dofCount <= 0)
        {
            return;
        }

        sum += Mathf.Abs(ab.jointVelocity[0]);
        count++;
    }

    // During the first few decision steps of an episode, joints are held at the rest pose instead of acting on the policy's output, letting physics and observations settle before the agent starts controlling the arm.
    private void HoldRestPoseDuringWarmup()
    {
        SetTarget(jointA1, _restA1);
        SetTarget(jointA2, _restA2);
        SetTarget(jointA3, _restA3);
        SetTarget(jointA4, _restA4);
        SetTarget(jointA5, _restA5);
        SetTarget(jointA6, _restA6);

        ZeroAllJointVelocities();
    }

    // Optional stability tweak: raises every joint drive's stiffness/damping/force limit up to a configured minimum, in case the imported robot's default drives are too weak to track targets reliably.
    private void StrengthenAllDrives()
    {
        StrengthenDrive(jointA1);
        StrengthenDrive(jointA2);
        StrengthenDrive(jointA3);
        StrengthenDrive(jointA4);
        StrengthenDrive(jointA5);
        StrengthenDrive(jointA6);
    }

    // Raises a single joint drive's stiffness/damping/force limit up to the configured minimums, leaving stronger existing values untouched.
    private void StrengthenDrive(ArticulationBody ab)
    {
        if (ab == null)
        {
            return;
        }

        var d = ab.xDrive;
        d.stiffness = Mathf.Max(d.stiffness, minDriveStiffness);
        d.damping = Mathf.Max(d.damping, minDriveDamping);
        d.forceLimit = Mathf.Max(d.forceLimit, minDriveForceLimit);
        ab.xDrive = d;
    }

    // Resets one joint to a target angle for episode start, optionally forcing the physical joint position (not just the drive target) and clearing its velocity, to guarantee a clean, repeatable starting state.
    private void ResetJointToSafePose(ArticulationBody ab, float targetDeg)
    {
        if (ab == null)
        {
            return;
        }

        SetTarget(ab, targetDeg);

        if (hardResetJointPositions && ab.jointPosition.dofCount > 0)
        {
            var jp = ab.jointPosition;
            jp[0] = Mathf.Clamp(targetDeg, jointMinAngle, jointMaxAngle) * Mathf.Deg2Rad;
            ab.jointPosition = jp;
        }

        if (zeroJointVelocityOnReset)
        {
            ZeroJointVelocity(ab);
        }
    }

    // Zeroes velocity across all six joints (used during the warmup hold).
    private void ZeroAllJointVelocities()
    {
        ZeroJointVelocity(jointA1);
        ZeroJointVelocity(jointA2);
        ZeroJointVelocity(jointA3);
        ZeroJointVelocity(jointA4);
        ZeroJointVelocity(jointA5);
        ZeroJointVelocity(jointA6);
    }

    // Clears a single joint's drive-axis velocity as well as its body-level linear/angular velocity.
    private void ZeroJointVelocity(ArticulationBody ab)
    {
        if (ab == null)
        {
            return;
        }

        if (ab.jointVelocity.dofCount > 0)
        {
            var v = ab.jointVelocity;
            v[0] = 0f;
            ab.jointVelocity = v;
        }

        ab.linearVelocity = Vector3.zero;
        ab.angularVelocity = Vector3.zero;
    }

    // Reads a joint's current physical angle if available, falling back to its drive target when the joint has no usable degree of freedom yet (e.g. before physics has initialised).
    private float GetCurrentAngleOrTarget(ArticulationBody ab)
    {
        if (ab == null)
        {
            return 0f;
        }

        if (ab.jointPosition.dofCount > 0)
        {
            return Mathf.Clamp(ab.jointPosition[0] * Mathf.Rad2Deg, jointMinAngle, jointMaxAngle);
        }

        return GetTarget(ab);
    }

    // Reads the joint drive's current target angle.
    private float GetTarget(ArticulationBody ab)
    {
        if (ab == null)
        {
            return 0f;
        }

        return ab.xDrive.target;
    }

    // Sets the joint drive's target angle, clamped to the configured joint limits.
    private void SetTarget(ArticulationBody ab, float target)
    {
        if (ab == null)
        {
            return;
        }

        var drive = ab.xDrive;
        drive.target = Mathf.Clamp(target, jointMinAngle, jointMaxAngle);
        ab.xDrive = drive;
    }

    // Sets the joint drive's target directly, without clamping to the arm's angle limits - used for the gripper fingers, whose open/close range is intentionally outside that range.
    private void SetRawTarget(ArticulationBody ab, float target)
    {
        if (ab == null)
        {
            return;
        }

        var drive = ab.xDrive;
        drive.target = target;
        ab.xDrive = drive;
    }

    // Applies an incremental change to a joint's drive target, clamped to the configured joint limits.
    // This is the core mechanism behind the action space.
    private void AddDeltaTarget(ArticulationBody ab, float delta)
    {
        if (ab == null)
        {
            return;
        }

        var drive = ab.xDrive;
        drive.target = Mathf.Clamp(drive.target + delta, jointMinAngle, jointMaxAngle);
        ab.xDrive = drive;
    }

    // Normalises a joint's current angle into roughly [-1, 1] for the observation vector.
    private float NormAngle(ArticulationBody ab)
    {
        if (ab == null || ab.jointPosition.dofCount <= 0)
        {
            return 0f;
        }

        float angle = ab.jointPosition[0] * Mathf.Rad2Deg;
        return Mathf.Clamp(angle / 180f, -1f, 1f);
    }

    // Clamps each component of a vector to [-5, 5], keeping observation values in a bounded, network-friendly range.
    private Vector3 ClampVec(Vector3 v)
    {
        return new Vector3(
            Mathf.Clamp(v.x, -5f, 5f),
            Mathf.Clamp(v.y, -5f, 5f),
            Mathf.Clamp(v.z, -5f, 5f)
        );
    }

    // Domain randomisation: samples a random position within the spawn area, rejecting candidates too close to the robot base, supporting generalisation across object positions as required by the brief.
    private Vector3 GetValidSpawnPosition()
    {
        Vector3 robotPos = robotBase != null ? robotBase.position : Vector3.zero;

        for (int i = 0; i < spawnMaxTries; i++)
        {
            float rx = Random.Range(-spawnHalfSize, spawnHalfSize);
            float rz = Random.Range(-spawnHalfSize, spawnHalfSize);

            Vector3 candidate = new Vector3(
                spawnCenter.x + rx,
                spawnCenter.y,
                spawnCenter.z + rz
            );

            float dxz = Vector2.Distance(
                new Vector2(candidate.x, candidate.z),
                new Vector2(robotPos.x, robotPos.z)
            );

            if (dxz >= forbiddenRadius)
            {
                return candidate;
            }
        }

        return spawnCenter + new Vector3(spawnHalfSize * 0.5f, 0f, 0f);
    }

    // Randomises the safe placement zone's position for this episode, wiring up the target object / robot base references if they have not already been set, supporting generalisation testing on the Placement goal.
    private void RandomizePlacementZone()
    {
        if (safePlacementZone == null)
        {
            return;
        }

        if (safePlacementZone.targetObject == null && targetObject != null)
        {
            safePlacementZone.targetObject = targetObject;
        }

        if (safePlacementZone.robotBase == null && robotBase != null)
        {
            safePlacementZone.robotBase = robotBase;
        }

        safePlacementZone.RandomizePosition();
        targetPlacement = safePlacementZone.transform;
    }

    // Returns the Grasp-and-Transport/Placement goal: a fixed offset above the safe placement zone, or a fallback position if the zone/placement transform is missing.
    private Vector3 GetPlacementGoalPosition()
    {
        if (safePlacementZone != null)
        {
            return safePlacementZone.transform.position + placementGoalOffset;
        }

        if (targetPlacement != null)
        {
            return targetPlacement.position + placementGoalOffset;
        }

        return spawnCenter + new Vector3(0.22f, desiredCarryHeightAboveSpawn, 0f);
    }

    // Selects the goal relevant to the agent's current sub-task: the placement point once the object is attached (Grasp and Transport stage), otherwise the object itself (Approach stage).
    private Vector3 GetActiveGoalPosition()
    {
        if (_attached)
        {
            return GetPlacementGoalPosition();
        }

        return targetObject != null ? targetObject.position : spawnCenter;
    }

    // Distance from the gripper to the target object - the core Approach stage distance metric.
    private float GraspToTargetDist()
    {
        if (graspPoint == null || targetObject == null)
        {
            return 1f;
        }

        return Vector3.Distance(graspPoint.position, targetObject.position);
    }

    // Distance from the target object to the placement goal - the core distance metric for the Grasp and Transport stage.
    private float TransportGoalDist()
    {
        if (targetObject == null)
        {
            return 1f;
        }

        return Vector3.Distance(targetObject.position, GetPlacementGoalPosition());
    }

    // Height of the carried object above the spawn plane, used for the carry height reward/penalty and the unsafe-drop check.
    private float CarryHeightAboveSpawn()
    {
        if (targetObject == null)
        {
            return 0f;
        }

        return targetObject.position.y - spawnCenter.y;
    }

    // Editor-only debug visualisation of the approach/attach zones, the placement zone, the spawn area, and the forbidden radius around the robot base.
    // Has no effect on training.
    private void OnDrawGizmos()
    {
        if (!drawDebugGizmos)
        {
            return;
        }

        if (graspPoint != null && targetObject != null)
        {
            Gizmos.color = Color.red;
            Gizmos.DrawLine(graspPoint.position, targetObject.position);

            Gizmos.color = new Color(0f, 1f, 0f, 0.2f);
            Gizmos.DrawSphere(targetObject.position, attachDistance);

            Gizmos.color = new Color(0f, 0.5f, 1f, 0.15f);
            Gizmos.DrawSphere(targetObject.position, closeStartDistance);
        }

        Vector3 placeGoal = GetPlacementGoalPosition();

        Gizmos.color = new Color(1f, 0.3f, 1f, 0.25f);
        Gizmos.DrawSphere(placeGoal, placementRadius);

        Gizmos.color = new Color(1f, 0.3f, 1f, 0.75f);
        Gizmos.DrawWireSphere(placeGoal, placementRadius);

        Gizmos.color = new Color(1f, 1f, 0f, 0.3f);
        Gizmos.DrawWireCube(
            spawnCenter,
            new Vector3(spawnHalfSize * 2f, 0.02f, spawnHalfSize * 2f)
        );

        if (robotBase != null)
        {
            Gizmos.color = new Color(1f, 0f, 0f, 0.2f);
            Gizmos.DrawWireSphere(
                new Vector3(robotBase.position.x, spawnCenter.y, robotBase.position.z),
                forbiddenRadius
            );
        }
    }
}