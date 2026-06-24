using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Reflection;

// STEP4_Release.cs
//
// Stage 4 of the staged Pick-and-Place curriculum: Placement and release.
// This script keeps the previous Approach, Grasp, and Transport behaviour, then adds the final lowering and release stage.
// Once the object is attached and carried above the safe placement zone, the active goal switches to a lower release point near the centre of the board.
// The arm is rewarded for lowering the object carefully, staying centred over the placement zone, and releasing only when the object is low and stable enough.
//
// The release is implemented as a safe snap release instead of immediately enabling normal object physics.
// After release, the target is detached from the gripper and held at a calculated safe placement position while remaining kinematic.
// This avoids the block being pushed away by sudden gravity, contact forces, or finger collisions after the arm has already reached a correct placement pose.
//
// Behavior Parameters:
// Behavior Name = NiryoReach
// Vector Observation Size = 20
// Continuous Actions = 6

public class Step4_Release : Agent
{
    // Six robot arm joints controlled by the RL policy.
    // The policy outputs one continuous action value for each joint.

    [Header("Robot Joints")]
    public ArticulationBody jointA1;
    public ArticulationBody jointA2;
    public ArticulationBody jointA3;
    public ArticulationBody jointA4;
    public ArticulationBody jointA5;
    public ArticulationBody jointA6;

    // Joint targets are clamped to keep the robot inside a reasonable motion range.
    [Header("Joint Limits")]
    public float jointMinAngle = -120f;
    public float jointMaxAngle = 120f;

    // The grasp point represents the end-effector used for distance measurements and scripted attachment.
    [Header("End Point")]
    public Transform graspPoint;

    // Scene references for the object, placement target, safe placement zone, and robot base.
    // The robot base is also used to avoid invalid spawn positions too close to the arm.
    [Header("Scene")]
    public Transform targetObject;
    public Transform targetPlacement;
    public Safeplacement safePlacementZone;
    public Transform robotBase;

    // Optional gripper controller and finger joints.
    // Gripper opening and closing is scripted so the learned actions can focus on arm movement.
    [Header("Gripper")]
    public MonoBehaviour gripperController;
    public ArticulationBody leftFingerJoint;
    public ArticulationBody rightFingerJoint;

    public float leftFingerOpenTarget = 0.01f;
    public float leftFingerClosedTarget = -0.01f;
    public float rightFingerOpenTarget = -0.01f;
    public float rightFingerClosedTarget = 0.01f;
    public float gripperCloseSpeed = 0.16f;

    // Random target spawning supports more varied training episodes.
    // Candidates too close to the robot base are rejected.
    [Header("Spawn Area")]
    public Vector3 spawnCenter = new Vector3(0.216f, 0.70f, -0.157f);
    public float spawnHalfSize = 0.10f;
    public float forbiddenRadius = 0.15f;
    public int spawnMaxTries = 20;

    // Movement parameters scale and smooth the continuous joint actions before applying them to drive targets.
    [Header("Movement")]
    public float jointSpeed = 0.95f;
    [Range(0.01f, 1f)] public float actionSmoothing = 0.22f;
    [Range(0.05f, 1f)] public float maxAbsAction = 0.65f;

    // Optional drive strengthening helps the imported articulation track joint targets more reliably.
    [Header("Drive Strength")]
    public bool strengthenJointDrives = true;
    public float minDriveStiffness = 18000f;
    public float minDriveDamping = 4200f;
    public float minDriveForceLimit = 1200f;

    // Reset and warmup options reduce unstable motion at the beginning of each episode.
    [Header("Reset")]
    public bool hardResetJointPositions = true;
    public bool zeroJointVelocityOnReset = true;
    public int warmupDecisionSteps = 3;

    // Maximum episode length before the attempt is treated as a timeout.
    [Header("Episode")]
    public int maxEpisodeSteps = 6800;

    // Approach and attachment thresholds inherited from the grasp stage.
    // These decide when the gripper closes and when the object becomes attached to the grasp point.
    [Header("Step2 Attach")]
    public float closeStartDistance = 0.12f;
    public float attachDistance = 0.072f;
    public int attachConfirmSteps = 1;
    public Vector3 attachedLocalPosition = Vector3.zero;
    public Vector3 attachedLocalEuler = Vector3.zero;

    // Transport thresholds define the upper goal above the safe placement zone.
    // The object must reach and hold this area before the lowering stage begins.
    [Header("Step3 Transport")]
    public Vector3 placementGoalOffset = new Vector3(0f, 0.105f, 0f);
    public float placementRadius = 0.145f;
    public float placementXZRadius = 0.185f;
    public int placementHoldSteps = 1;

    // Lowering thresholds define the final pre-release target.
    // The agent should stay centred, move low enough to place the object, and avoid lowering too far.
    [Header("Step4 Lower")]
    public float lowerGoalHeightAboveZone = 0.035f;
    public float lowerReleaseRadius = 0.105f;
    public float releaseTriggerHeightAboveZone = 0.10f;
    public float releaseMinHeightAboveZone = -0.03f;
    public float releaseStrictXZRadius = 0.125f;
    public float maxLowerJointVelocityForRelease = 20.0f;
    public int lowerHoldSteps = 1;
    public int lowerForceReleaseAfterSteps = 22;

    // Safe-release settings check whether the object remains inside the placement zone after release.
    // Success requires the object to stay within tolerance for several consecutive steps.
    [Header("Safe Release")]
    public int releaseSuccessHoldSteps = 5;
    public int releaseMaxWaitSteps = 45;
    public float releaseZoneTolerance = 0.05f;
    public float landedMaxHeightAboveZone = 0.115f;

    // Carry-height settings encourage the arm to lift the object before moving horizontally.
    // Carrying too low is penalised because it often causes unstable transport.
    [Header("Carry Height")]
    public float minCarryHeightAboveSpawn = 0.05f;
    public float desiredCarryHeightAboveSpawn = 0.11f;
    public float maxCarryHeightAboveSpawn = 0.36f;
    public int lowCarryStepLimit = 220;

    // Safety thresholds terminate episodes that leave the useful workspace or become unstable.
    [Header("Safety")]
    public float minAllowedGraspPointY = 0.28f;
    public float maxWorkspaceDistanceFromSpawnCenter = 1.25f;
    public float maxAverageJointVelocity = 10f;
    public int unsafeVelocityStepLimit = 12;

    // Reward shaping for the approach and attachment stage.
    // The agent is rewarded for moving closer, entering the close zone, becoming ready to attach, and attaching successfully.
    [Header("Reward — Approach / Attach")]
    public float approachImprovementScale = 3.0f;
    public float approachDistancePenaltyScale = 0.0015f;
    public float closeZoneReward = 0.002f;
    public float attachReadyReward = 0.010f;
    public float attachBonus = 1.0f;

    // Reward shaping for the transport stage.
    // The agent is rewarded for stable grasping, lifting, moving toward the placement area, and centering over the board.
    [Header("Reward — Transport")]
    public float transportImprovementScale = 2.2f;
    public float transportDistancePenaltyScale = 0.0015f;
    public float carryHeightReward = 0.003f;
    public float lowCarryPenalty = -0.0015f;
    public float placeZoneReward = 0.015f;
    public float boardCenterReward = 0.020f;
    public float lockReward = 0.0010f;
    public float lockErrorPenaltyScale = 0.006f;

    // Reward shaping for the lowering stage.
    // These rewards guide the object toward a low, centred, and stable release pose.
    [Header("Reward — Lower")]
    public float lowerStartBonus = 1.2f;
    public float lowerImprovementScale = 6.0f;
    public float lowerDistancePenaltyScale = 0.002f;
    public float lowerZoneReward = 0.040f;
    public float lowEnoughReward = 0.100f;
    public float lowerStableReward = 0.010f;
    public float lowerCenteringReward = 0.050f;
    public float tooHighBeforeReleasePenalty = -0.004f;
    public float tooLowBeforeReleasePenalty = -0.006f;
    public float outsideBoardPenalty = -0.003f;

    // Reward shaping for the final placement check.
    // The largest reward is only given when the released object stays stable inside the placement zone.
    [Header("Reward — Release")]
    public float releaseStartBonus = 6.0f;
    public float releaseInZoneReward = 0.22f;
    public float releasedStableReward = 0.25f;
    public float releasedHeightReward = 0.22f;
    public float releaseSuccessReward = 45.0f;
    public float releaseFailPenalty = -2.0f;
    public float noReleaseTimeoutPenalty = -1.0f;

    // Shared penalties encourage efficient, smooth, and safe movement across all stages.
    [Header("Common Reward / Penalty")]
    public float timePenalty = -0.00005f;
    public float stuckPenalty = -0.001f;
    public float timeoutPenalty = -0.5f;
    public float unsafePenalty = -1.0f;
    public float jointVelocityPenaltyScale = 0.0015f;
    public float actionChangePenaltyScale = 0.0012f;

    // Stuck detection penalises long periods with no meaningful progress.
    [Header("Stuck Detection")]
    public float minImprovement = 0.0003f;
    public int stuckStepLimit = 900;

    // Debug gizmos visualise the active target areas and placement thresholds in the Unity Scene view.
    [Header("Debug")]
    public bool drawDebugGizmos = true;

    // Initial joint targets used to reset the robot at the start of each episode.
    private float _restA1;
    private float _restA2;
    private float _restA3;
    private float _restA4;
    private float _restA5;
    private float _restA6;

    // Distance tracking variables used for progress rewards in approach, transport, and lowering.
    private float _prevReachDist;
    private float _bestReachDist;
    private float _prevTransportDist;
    private float _bestTransportDist;
    private float _prevLowerDist;
    private float _bestLowerDist;

    // Episode counters for timeout, stuck detection, safety checks, stage holds, and release validation.
    private int _stepCount;
    private int _stuckSteps;
    private int _unsafeVelocitySteps;
    private int _attachZoneSteps;
    private int _placementHoldCounter;
    private int _lowerHoldCounter;
    private int _lowerStepCounter;
    private int _lowCarrySteps;
    private int _releaseStepCounter;
    private int _releaseSuccessCounter;

    // Stage flags.
    // The active goal and reward logic change depending on whether the object is attached, lowering, or released.
    private bool _attached;
    private bool _lowering;
    private bool _released;

    // Normalised scripted gripper state, where 0 is open and 1 is closed.
    private float _gripperClose01;

    // Target physics and parenting state used for reset, attachment, transport, and release.
    private Rigidbody _targetRb;
    private Transform _targetOriginalParent;
    private Quaternion _targetOriginalRotation;

    // Frozen release pose used by the safe snap release after the object is detached.
    private Vector3 _releaseFrozenPosition;
    private Quaternion _releaseFrozenRotation;

    // Action smoothing buffers used to reduce sudden joint movement.
    private readonly float[] _smoothedActions = new float[6];
    private readonly float[] _lastAppliedActions = new float[6];

    // Position normalisation scale for vector observations.
    private const float OBS_NORM = 0.5f;

    // One-time setup when the agent is created: cache joint rest angles, store target reset references,
    // connect the placement transform if needed, and disable target physics for the scripted grasp/release pipeline.
    public override void Initialize()
    {
        ForcePreset();
        Debug.Log("[STEP4_RELEASE_SAFE_SNAP_V6_LOADED]");

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

        PrepareTargetPhysicsOff();
    }

    // Resets everything needed for a new episode: joints return to their rest pose, the gripper opens,
    // the target object is re-spawned, the placement zone is randomised, and all stage trackers are cleared.
    public override void OnEpisodeBegin()
    {
        ForcePreset();

        SetRobotTargetCollisionsIgnored(false);

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

        _stepCount = 0;
        _stuckSteps = 0;
        _unsafeVelocitySteps = 0;
        _attachZoneSteps = 0;
        _placementHoldCounter = 0;
        _lowerHoldCounter = 0;
        _lowerStepCounter = 0;
        _lowCarrySteps = 0;
        _releaseStepCounter = 0;
        _releaseSuccessCounter = 0;

        _attached = false;
        _lowering = false;
        _released = false;
        _gripperClose01 = 0f;

        OpenGripperImmediately();
        ResetTargetObject();
        RandomizePlacementZone();

        _prevReachDist = GraspToTargetDist();
        _bestReachDist = _prevReachDist;

        _prevTransportDist = TransportGoalDist();
        _bestTransportDist = _prevTransportDist;

        _prevLowerDist = LowerGoalDist();
        _bestLowerDist = _prevLowerDist;
    }

    // Re-applies the chosen Step4 training preset.
    // This keeps movement limits, reward scales, safety thresholds, and release tolerances consistent between episodes.
    private void ForcePreset()
    {
        jointSpeed = 0.95f;
        actionSmoothing = 0.22f;
        maxAbsAction = 0.65f;

        closeStartDistance = 0.12f;
        attachDistance = 0.072f;
        attachConfirmSteps = 1;

        placementGoalOffset = new Vector3(0f, 0.105f, 0f);
        placementRadius = 0.145f;
        placementXZRadius = 0.185f;
        placementHoldSteps = 1;

        lowerGoalHeightAboveZone = 0.035f;
        lowerReleaseRadius = 0.105f;
        releaseTriggerHeightAboveZone = 0.10f;
        releaseMinHeightAboveZone = -0.03f;
        releaseStrictXZRadius = 0.125f;
        maxLowerJointVelocityForRelease = 20.0f;
        lowerHoldSteps = 1;
        lowerForceReleaseAfterSteps = 22;

        releaseSuccessHoldSteps = 5;
        releaseMaxWaitSteps = 45;
        releaseZoneTolerance = 0.05f;
        landedMaxHeightAboveZone = 0.115f;

        minCarryHeightAboveSpawn = 0.05f;
        desiredCarryHeightAboveSpawn = 0.11f;
        maxCarryHeightAboveSpawn = 0.36f;
        lowCarryStepLimit = 220;

        maxAverageJointVelocity = 10.0f;
        unsafeVelocityStepLimit = 12;

        approachImprovementScale = 3.0f;
        approachDistancePenaltyScale = 0.0015f;
        closeZoneReward = 0.002f;
        attachReadyReward = 0.010f;
        attachBonus = 1.0f;

        transportImprovementScale = 2.2f;
        transportDistancePenaltyScale = 0.0015f;
        carryHeightReward = 0.003f;
        lowCarryPenalty = -0.0015f;
        placeZoneReward = 0.015f;
        boardCenterReward = 0.020f;
        lockReward = 0.0010f;
        lockErrorPenaltyScale = 0.006f;

        lowerStartBonus = 1.2f;
        lowerImprovementScale = 6.0f;
        lowerDistancePenaltyScale = 0.002f;
        lowerZoneReward = 0.040f;
        lowEnoughReward = 0.100f;
        lowerStableReward = 0.010f;
        lowerCenteringReward = 0.050f;
        tooHighBeforeReleasePenalty = -0.004f;
        tooLowBeforeReleasePenalty = -0.006f;
        outsideBoardPenalty = -0.003f;

        releaseStartBonus = 6.0f;
        releaseInZoneReward = 0.22f;
        releasedStableReward = 0.25f;
        releasedHeightReward = 0.22f;
        releaseSuccessReward = 45.0f;
        releaseFailPenalty = -2.0f;
        noReleaseTimeoutPenalty = -1.0f;

        timePenalty = -0.00005f;
        stuckPenalty = -0.001f;
        timeoutPenalty = -0.5f;
        unsafePenalty = -1.0f;
        jointVelocityPenaltyScale = 0.0015f;
        actionChangePenaltyScale = 0.0012f;
        minImprovement = 0.0003f;
        stuckStepLimit = 900;

        strengthenJointDrives = true;
        minDriveStiffness = 18000f;
        minDriveDamping = 4200f;
        minDriveForceLimit = 1200f;
    }

    // Observation space for the full Step4 task: six normalised joint angles, grasp point position,
    // the current active goal, vector and distance from gripper to goal, height, episode progress,
    // stage state, and best distance reached on the current sub-task.
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(NormAngle(jointA1));
        sensor.AddObservation(NormAngle(jointA2));
        sensor.AddObservation(NormAngle(jointA3));
        sensor.AddObservation(NormAngle(jointA4));
        sensor.AddObservation(NormAngle(jointA5));
        sensor.AddObservation(NormAngle(jointA6));

        Vector3 gpRel = graspPoint != null ? (graspPoint.position - spawnCenter) / OBS_NORM : Vector3.zero;
        sensor.AddObservation(ClampVec(gpRel));

        Vector3 activeGoal = GetActiveGoalPosition();
        Vector3 activeGoalRel = (activeGoal - spawnCenter) / OBS_NORM;
        sensor.AddObservation(ClampVec(activeGoalRel));

        Vector3 toActiveGoal = graspPoint != null ? activeGoal - graspPoint.position : Vector3.zero;
        sensor.AddObservation(ClampVec(toActiveGoal));

        sensor.AddObservation(Mathf.Clamp(toActiveGoal.magnitude, 0f, 5f));
        sensor.AddObservation(Mathf.Clamp(graspPoint != null ? graspPoint.position.y - spawnCenter.y : 0f, -5f, 5f));
        sensor.AddObservation((float)_stepCount / Mathf.Max(1, maxEpisodeSteps));

        float stageObs = 0f;

        if (_released)
        {
            stageObs = -1f;
        }
        else if (_lowering)
        {
            stageObs = 0.5f;
        }
        else if (_attached)
        {
            stageObs = 1f;
        }
        else
        {
            stageObs = _gripperClose01;
        }

        sensor.AddObservation(stageObs);

        float progressObs = _released
            ? ReleaseXZDistance()
            : (_lowering ? _bestLowerDist : (_attached ? _bestTransportDist : _bestReachDist));

        sensor.AddObservation(Mathf.Clamp01(progressObs));
    }

    // Action space and main control loop: six continuous joint deltas are applied through ApplySafeJointActions.
    // After a short warmup, the agent runs Approach until attached, Transport until above the board,
    // Lower until the object is ready to release, and then validates the final safe placement.
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
            _prevLowerDist = LowerGoalDist();
            return;
        }

        if (_released)
        {
            HoldCurrentJointTargets();
            UpdateScriptedGripper();
            StepSafeReleasePlacement();
        }
        else
        {
            ApplySafeJointActions(actions);
            UpdateScriptedGripper();

            if (!_attached)
            {
                StepApproachAndAttach();
            }
            else if (!_lowering)
            {
                StepTransportToPlacementUpper();
            }
            else
            {
                StepLowerBeforeRelease();
            }
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
            if (!_released)
            {
                AddReward(noReleaseTimeoutPenalty);
            }
            else
            {
                AddReward(timeoutPenalty);
            }

            EndEpisode();
        }
    }

    // Manual keyboard control for testing the six joints without a trained policy.
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

    // Approach stage reward shaping with a simplified grasp trigger.
    // The agent is rewarded for reducing gripper-to-object distance and entering the close/attach zones.
    // Once the attach condition is held long enough, the object is parented to the gripper to simulate a stable grasp.
    private void StepApproachAndAttach()
    {
        float currReachDist = GraspToTargetDist();

        AddProgressReward(_prevReachDist, currReachDist, approachImprovementScale, 0.0015f, 0.08f);
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

        if (_attachZoneSteps >= attachConfirmSteps)
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

    // Grasp and Transport stage reward shaping.
    // The object is kept kinematically locked to the gripper while the agent is rewarded for carrying it at a safe height,
    // reducing distance to the placement goal, and centering over the board before lowering.
    private void StepTransportToPlacementUpper()
    {
        KeepTargetLockedToGraspPoint();

        float currTransportDist = TransportGoalDist();
        float xzDist = ReleaseXZDistance();
        float heightAboveSpawn = CarryHeightAboveSpawn();

        float lockError = targetObject != null && graspPoint != null
            ? Vector3.Distance(targetObject.position, graspPoint.position)
            : 1f;

        AddReward(lockReward);
        AddReward(-lockError * lockErrorPenaltyScale);
        AddReward(timePenalty);

        float lift01 = Mathf.InverseLerp(minCarryHeightAboveSpawn, desiredCarryHeightAboveSpawn, heightAboveSpawn);
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

        AddProgressReward(_prevTransportDist, currTransportDist, transportImprovementScale * Mathf.Lerp(0.4f, 1f, lift01), 0.002f, 0.10f);
        AddReward(-currTransportDist * transportDistancePenaltyScale * Time.fixedDeltaTime);

        if (xzDist <= releaseStrictXZRadius)
        {
            AddReward(boardCenterReward);
        }

        if (currTransportDist < _bestTransportDist)
        {
            _bestTransportDist = currTransportDist;
        }

        bool closeEnoughXZ = xzDist <= placementXZRadius;
        bool closeEnough3D = currTransportDist <= placementRadius;
        bool highEnough = heightAboveSpawn >= minCarryHeightAboveSpawn;

        if ((closeEnough3D || closeEnoughXZ) && highEnough)
        {
            _placementHoldCounter++;
            AddReward(placeZoneReward);

            if (_placementHoldCounter >= placementHoldSteps)
            {
                BeginLoweringBeforeRelease();
                return;
            }
        }
        else
        {
            _placementHoldCounter = 0;
        }

        if (_lowCarrySteps > lowCarryStepLimit)
        {
            AddReward(-0.3f);
            _lowCarrySteps = 0;
        }

        AddStuckPenalty(_prevTransportDist, currTransportDist);
        _prevTransportDist = currTransportDist;
    }

    // Starts the Placement stage once the object has reached the upper placement zone.
    // Lowering uses its own distance tracker so the reward focuses on moving down into the release window.
    private void BeginLoweringBeforeRelease()
    {
        _lowering = true;
        _lowerHoldCounter = 0;
        _lowerStepCounter = 0;

        _prevLowerDist = LowerGoalDist();
        _bestLowerDist = _prevLowerDist;

        AddReward(lowerStartBonus);
    }

    // Placement/lowering reward shaping.
    // The object should stay centred over the board, move low enough to be placed, avoid going below the safe range,
    // and keep joint velocity low before the final release is triggered.
    private void StepLowerBeforeRelease()
    {
        KeepTargetLockedToGraspPoint();
        _lowerStepCounter++;

        float currLowerDist = LowerGoalDist();
        float heightAboveZone = HeightAboveZone();
        float xzDist = ReleaseXZDistance();
        float avgJointVel = AverageJointVelocityAbs();

        AddReward(lockReward);
        AddReward(timePenalty);

        AddProgressReward(_prevLowerDist, currLowerDist, lowerImprovementScale, 0.002f, 0.16f);
        AddReward(-currLowerDist * lowerDistancePenaltyScale * Time.fixedDeltaTime);

        if (currLowerDist < _bestLowerDist)
        {
            _bestLowerDist = currLowerDist;
        }

        bool xzStrict = xzDist <= releaseStrictXZRadius;
        bool lowEnough = heightAboveZone <= releaseTriggerHeightAboveZone;
        bool notTooLow = heightAboveZone >= releaseMinHeightAboveZone;
        bool stableEnough = avgJointVel <= maxLowerJointVelocityForRelease;
        bool closeToLowerGoal = currLowerDist <= lowerReleaseRadius;

        if (xzStrict)
        {
            float center01 = Mathf.Clamp01(1f - xzDist / Mathf.Max(0.001f, releaseStrictXZRadius));
            AddReward(lowerZoneReward);
            AddReward(lowerCenteringReward * center01);
        }
        else
        {
            AddReward(outsideBoardPenalty);
        }

        if (lowEnough && notTooLow)
        {
            float low01 = Mathf.Clamp01(1f - Mathf.Abs(heightAboveZone - lowerGoalHeightAboveZone) / 0.10f);
            AddReward(lowEnoughReward * Mathf.Max(0.25f, low01));
        }
        else if (!lowEnough)
        {
            float highPenalty01 = Mathf.Clamp01((heightAboveZone - releaseTriggerHeightAboveZone) / 0.20f + 0.25f);
            AddReward(tooHighBeforeReleasePenalty * highPenalty01);
        }
        else
        {
            AddReward(tooLowBeforeReleasePenalty);
        }

        if (stableEnough)
        {
            AddReward(lowerStableReward);
        }
        else
        {
            AddReward(-0.002f);
        }

        bool normalReleaseReady = closeToLowerGoal && xzStrict && lowEnough && notTooLow;

        bool forceReleaseReady =
            _lowerStepCounter >= lowerForceReleaseAfterSteps &&
            xzDist <= placementXZRadius + 0.02f &&
            heightAboveZone <= releaseTriggerHeightAboveZone + 0.06f &&
            notTooLow;

        bool emergencyReleaseReady =
            _lowerStepCounter >= 45 &&
            xzDist <= placementXZRadius + 0.05f &&
            heightAboveZone <= releaseTriggerHeightAboveZone + 0.12f &&
            notTooLow;

        if (normalReleaseReady)
        {
            _lowerHoldCounter++;
            AddReward(0.12f);

            if (_lowerHoldCounter >= lowerHoldSteps)
            {
                AddReward(2.0f);
                BeginSafeReleaseAtPlacement();
                return;
            }
        }
        else
        {
            _lowerHoldCounter = 0;
        }

        if (forceReleaseReady || emergencyReleaseReady)
        {
            AddReward(forceReleaseReady ? 0.8f : 0.25f);
            BeginSafeReleaseAtPlacement();
            return;
        }

        AddStuckPenalty(_prevLowerDist, currLowerDist);
        _prevLowerDist = currLowerDist;
    }

    // Starts the safe snap release.
    // The object is detached from the gripper and moved to the calculated release pose, but stays kinematic.
    // This avoids a correct placement being ruined by sudden gravity, contact forces, or gripper collisions.
    private void BeginSafeReleaseAtPlacement()
    {
        if (targetObject == null)
        {
            return;
        }

        AddReward(releaseStartBonus);

        _released = true;
        _attached = false;
        _lowering = false;

        _releaseStepCounter = 0;
        _releaseSuccessCounter = 0;
        _gripperClose01 = 0f;

        HoldCurrentJointTargets();
        ZeroAllJointVelocities();

        targetObject.SetParent(_targetOriginalParent, true);

        _releaseFrozenPosition = GetSafeReleasePosition();
        _releaseFrozenRotation = targetObject.rotation;

        targetObject.position = _releaseFrozenPosition;
        targetObject.rotation = _releaseFrozenRotation;

        if (_targetRb == null)
        {
            _targetRb = targetObject.GetComponent<Rigidbody>();
        }

        if (_targetRb != null)
        {
            _targetRb.isKinematic = true;
            _targetRb.detectCollisions = false;
            _targetRb.useGravity = false;
        }

        SetTargetCollidersEnabled(false);
        SetRobotTargetCollisionsIgnored(true);
        OpenGripperImmediately();
    }

    // Final placement validation.
    // The object is held at the safe release pose while the script checks that it remains inside the placement tolerance
    // and at an acceptable height for enough consecutive steps to count as a successful placement.
    private void StepSafeReleasePlacement()
    {
        _releaseStepCounter++;
        AddReward(timePenalty);

        if (targetObject == null)
        {
            AddReward(releaseFailPenalty);
            EndEpisode();
            return;
        }

        _releaseFrozenPosition = GetSafeReleasePosition();
        targetObject.position = _releaseFrozenPosition;
        targetObject.rotation = _releaseFrozenRotation;

        if (_targetRb == null)
        {
            _targetRb = targetObject.GetComponent<Rigidbody>();
        }

        if (_targetRb != null)
        {
            _targetRb.isKinematic = true;
            _targetRb.detectCollisions = false;
            _targetRb.useGravity = false;
        }

        SetTargetCollidersEnabled(false);
        OpenGripperImmediately();

        float xzDist = ReleaseXZDistance();
        float heightAboveZone = HeightAboveZone();

        bool insideXZ = xzDist <= releaseStrictXZRadius + releaseZoneTolerance;
        bool goodHeight = heightAboveZone >= releaseMinHeightAboveZone - 0.04f &&
                          heightAboveZone <= landedMaxHeightAboveZone;

        if (insideXZ)
        {
            AddReward(releaseInZoneReward);
        }

        if (goodHeight)
        {
            AddReward(releasedHeightReward);
        }

        if (insideXZ && goodHeight)
        {
            _releaseSuccessCounter++;
            AddReward(releasedStableReward);

            if (_releaseSuccessCounter >= releaseSuccessHoldSteps)
            {
                AddReward(releaseSuccessReward);
                EndEpisode();
                return;
            }
        }
        else
        {
            _releaseSuccessCounter = 0;
        }

        if (_releaseStepCounter >= releaseMaxWaitSteps)
        {
            if (insideXZ && goodHeight)
            {
                AddReward(releaseSuccessReward * 0.5f);
            }
            else
            {
                AddReward(releaseFailPenalty);
            }

            EndEpisode();
        }
    }

    // Converts the policy's six continuous actions into incremental joint-drive targets.
    // Action magnitude is reduced near grasping, centering, and lowering to encourage fine control and reduce overshoot.
    private void ApplySafeJointActions(ActionBuffers actions)
    {
        float scale = 1f;

        if (!_attached)
        {
            float dist = GraspToTargetDist();

            if (dist < closeStartDistance)
            {
                scale = 0.72f;
            }

            if (dist < attachDistance * 1.4f)
            {
                scale = 0.50f;
            }
        }
        else if (_lowering)
        {
            scale = 0.62f;
        }
        else
        {
            float heightAbove = CarryHeightAboveSpawn();
            scale = heightAbove < minCarryHeightAboveSpawn ? 0.74f : 0.58f;

            if (ReleaseXZDistance() < placementXZRadius)
            {
                scale *= 0.82f;
            }
        }

        AddDeltaTarget(jointA1, SmoothAction(actions.ContinuousActions[0], 0) * jointSpeed * 1.00f * scale);
        AddDeltaTarget(jointA2, SmoothAction(actions.ContinuousActions[1], 1) * jointSpeed * 0.88f * scale);
        AddDeltaTarget(jointA3, SmoothAction(actions.ContinuousActions[2], 2) * jointSpeed * 0.82f * scale);
        AddDeltaTarget(jointA4, SmoothAction(actions.ContinuousActions[3], 3) * jointSpeed * 0.52f * scale);
        AddDeltaTarget(jointA5, SmoothAction(actions.ContinuousActions[4], 4) * jointSpeed * 0.42f * scale);
        AddDeltaTarget(jointA6, SmoothAction(actions.ContinuousActions[5], 5) * jointSpeed * 0.36f * scale);
    }

    // Clamps a raw action to the allowed range and low-pass filters it against the previous value.
    // This reduces jitter in the commanded joint motion.
    private float SmoothAction(float rawAction, int index)
    {
        float clamped = Mathf.Clamp(rawAction, -maxAbsAction, maxAbsAction);
        _smoothedActions[index] = Mathf.Lerp(_smoothedActions[index], clamped, actionSmoothing);
        return _smoothedActions[index];
    }

    // Gripper open/close is scripted rather than learned.
    // The gripper stays open before attachment, closes while carrying, and opens again during release.
    private void UpdateScriptedGripper()
    {
        float targetClose;

        if (_released)
        {
            targetClose = 0f;
        }
        else if (_attached)
        {
            targetClose = 1f;
        }
        else
        {
            targetClose = 0f;
        }

        float speed = _released ? gripperCloseSpeed * 2.5f : gripperCloseSpeed * 1.2f;

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

    // Forces the gripper fully open at the start of an episode or during release.
    private void OpenGripperImmediately()
    {
        _gripperClose01 = 0f;
        SetRawTarget(leftFingerJoint, leftFingerOpenTarget);
        SetRawTarget(rightFingerJoint, rightFingerOpenTarget);
        DriveOptionalGripperController(0f);
    }

    // Optional hook for an external gripper controller component.
    // This keeps the script compatible with scenes that use different gripper implementations.
    private void DriveOptionalGripperController(float close01)
    {
        if (gripperController == null)
        {
            return;
        }

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

    // Reflection helper for optional parameterless gripper methods.
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

    // Restores the target to its original parent, randomises its spawn position, and disables unstable physics.
    private void ResetTargetObject()
    {
        if (targetObject == null)
        {
            return;
        }

        targetObject.SetParent(_targetOriginalParent, true);
        targetObject.position = GetValidSpawnPosition();
        targetObject.rotation = _targetOriginalRotation;

        PrepareTargetPhysicsOff();
    }

    // Simplifies target physics before grasping, while carrying, and after release.
    // Gravity, collision detection, and colliders are disabled so the object is controlled by the staged task logic.
    private void PrepareTargetPhysicsOff()
    {
        if (targetObject == null)
        {
            return;
        }

        if (_targetRb == null)
        {
            _targetRb = targetObject.GetComponent<Rigidbody>();
        }

        if (_targetRb != null)
        {
            _targetRb.isKinematic = true;
            _targetRb.detectCollisions = false;
            _targetRb.useGravity = false;
        }

        SetTargetCollidersEnabled(false);
    }

    // Implements the simplified grasp.
    // Once the approach stage succeeds, the target is parented to the grasp point at a fixed local offset,
    // guaranteeing a stable hold while the arm transports and lowers it.
    private void AttachTargetToGraspPoint()
    {
        if (targetObject == null || graspPoint == null)
        {
            return;
        }

        _attached = true;
        _lowering = false;
        _released = false;

        _placementHoldCounter = 0;
        _lowerHoldCounter = 0;
        _lowerStepCounter = 0;
        _lowCarrySteps = 0;

        if (_targetRb == null)
        {
            _targetRb = targetObject.GetComponent<Rigidbody>();
        }

        if (_targetRb != null)
        {
            _targetRb.isKinematic = true;
            _targetRb.detectCollisions = false;
            _targetRb.useGravity = false;
        }

        SetTargetCollidersEnabled(false);

        targetObject.SetParent(graspPoint, false);
        targetObject.localPosition = attachedLocalPosition;
        targetObject.localRotation = Quaternion.Euler(attachedLocalEuler);
    }

    // Maintains the simplified grasp every step while the object is attached.
    // This prevents small transform or parenting drift from affecting the transport and lowering rewards.
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

        if (_targetRb == null)
        {
            _targetRb = targetObject.GetComponent<Rigidbody>();
        }

        if (_targetRb != null)
        {
            _targetRb.isKinematic = true;
            _targetRb.detectCollisions = false;
            _targetRb.useGravity = false;
        }

        SetTargetCollidersEnabled(false);
    }

    // Enables or disables all colliders on the target object.
    private void SetTargetCollidersEnabled(bool enabled)
    {
        if (targetObject == null)
        {
            return;
        }

        Collider[] cols = targetObject.GetComponentsInChildren<Collider>(true);

        foreach (Collider c in cols)
        {
            if (c != null)
            {
                c.enabled = enabled;
            }
        }
    }

    // Ignores or restores collisions between the robot and target during the safe release phase.
    private void SetRobotTargetCollisionsIgnored(bool ignored)
    {
        if (targetObject == null || robotBase == null)
        {
            return;
        }

        Collider[] targetCols = targetObject.GetComponentsInChildren<Collider>(true);
        Collider[] robotCols = robotBase.GetComponentsInChildren<Collider>(true);

        foreach (Collider tc in targetCols)
        {
            if (tc == null)
            {
                continue;
            }

            foreach (Collider rc in robotCols)
            {
                if (rc == null || rc == tc)
                {
                    continue;
                }

                Physics.IgnoreCollision(tc, rc, ignored);
            }
        }
    }

    // Shared progress reward used across approach, transport, and lowering.
    // Moving closer to the current goal gives positive reward; moving away receives a small capped penalty.
    private void AddProgressReward(float prevDist, float currDist, float positiveScale, float backtrackPenalty, float positiveCap)
    {
        float delta = prevDist - currDist;

        if (delta > 0f)
        {
            AddReward(Mathf.Min(delta * positiveScale, positiveCap));
        }
        else
        {
            AddReward(Mathf.Max(delta * positiveScale * 0.04f, -backtrackPenalty));
        }
    }

    // Tracks consecutive steps without meaningful progress and applies a penalty after the stuck limit.
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

    // Regularises the action space by penalising abrupt action changes and excessive joint velocity.
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

    // Safety and early-termination check.
    // Flags NaN positions, gripper height below the safe limit, workspace escape, excessive lowering,
    // or sustained high joint velocity.
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

        if (!_released && p.y < minAllowedGraspPointY)
        {
            return true;
        }

        if (Vector3.Distance(p, spawnCenter) > maxWorkspaceDistanceFromSpawnCenter)
        {
            return true;
        }

        if (!_released && _attached && HeightAboveZone() < releaseMinHeightAboveZone - 0.08f)
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

    // Average absolute joint velocity across all six controlled joints.
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

    // Adds one joint's absolute velocity into the running average calculation.
    private void AddJointVelocity(ArticulationBody ab, ref float sum, ref int count)
    {
        if (ab == null || ab.jointVelocity.dofCount <= 0)
        {
            return;
        }

        sum += Mathf.Abs(ab.jointVelocity[0]);
        count++;
    }

    // During the first few decision steps, hold the rest pose so physics and observations settle.
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

    // Holds the current joint targets after release so the arm does not keep moving.
    private void HoldCurrentJointTargets()
    {
        SetTarget(jointA1, GetTarget(jointA1));
        SetTarget(jointA2, GetTarget(jointA2));
        SetTarget(jointA3, GetTarget(jointA3));
        SetTarget(jointA4, GetTarget(jointA4));
        SetTarget(jointA5, GetTarget(jointA5));
        SetTarget(jointA6, GetTarget(jointA6));
    }

    // Applies the optional drive-strengthening setup to all six robot joints.
    private void StrengthenAllDrives()
    {
        StrengthenDrive(jointA1);
        StrengthenDrive(jointA2);
        StrengthenDrive(jointA3);
        StrengthenDrive(jointA4);
        StrengthenDrive(jointA5);
        StrengthenDrive(jointA6);
    }

    // Raises one joint drive's stiffness, damping, and force limit up to the configured minimums.
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

    // Resets one joint to the stored rest target and optionally clears its physical position and velocity.
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

    // Clears velocity across all six controlled joints.
    private void ZeroAllJointVelocities()
    {
        ZeroJointVelocity(jointA1);
        ZeroJointVelocity(jointA2);
        ZeroJointVelocity(jointA3);
        ZeroJointVelocity(jointA4);
        ZeroJointVelocity(jointA5);
        ZeroJointVelocity(jointA6);
    }

    // Clears one joint's drive-axis velocity and body-level linear/angular velocity.
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

    // Reads a joint's current physical angle if available, otherwise falls back to its drive target.
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

    // Reads the current xDrive target angle of a joint.
    private float GetTarget(ArticulationBody ab)
    {
        if (ab == null)
        {
            return 0f;
        }

        return ab.xDrive.target;
    }

    // Sets a joint drive target while clamping it to the configured joint limits.
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

    // Sets a drive target directly without arm-joint clamping, used for the gripper fingers.
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

    // Applies an incremental joint target update generated by the policy action.
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

    // Normalises a joint angle into roughly [-1, 1] for the observation vector.
    private float NormAngle(ArticulationBody ab)
    {
        if (ab == null || ab.jointPosition.dofCount <= 0)
        {
            return 0f;
        }

        float angle = ab.jointPosition[0] * Mathf.Rad2Deg;
        return Mathf.Clamp(angle / 180f, -1f, 1f);
    }

    // Clamps vector observations to a bounded range for more stable neural network inputs.
    private Vector3 ClampVec(Vector3 v)
    {
        return new Vector3(
            Mathf.Clamp(v.x, -5f, 5f),
            Mathf.Clamp(v.y, -5f, 5f),
            Mathf.Clamp(v.z, -5f, 5f)
        );
    }

    // Samples a random target position within the spawn area, rejecting candidates too close to the robot base.
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

    // Randomises the safe placement zone and wires up target/base references if they have not already been set.
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

    // Returns the upper transport goal above the safe placement zone.
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

    // Returns the lower pre-release goal near the centre of the placement zone.
    private Vector3 GetLowerReleaseGoalPosition()
    {
        Vector3 zone = targetPlacement != null ? targetPlacement.position : spawnCenter;

        if (safePlacementZone != null)
        {
            zone = safePlacementZone.transform.position;
        }

        return new Vector3(
            zone.x,
            zone.y + lowerGoalHeightAboveZone,
            zone.z
        );
    }

    // Returns the final safe snap position used after the object is detached from the gripper.
    private Vector3 GetSafeReleasePosition()
    {
        Vector3 zone = targetPlacement != null ? targetPlacement.position : GetLowerReleaseGoalPosition();

        if (safePlacementZone != null)
        {
            zone = safePlacementZone.transform.position;
        }

        float safeY = GetZoneBaseY() + lowerGoalHeightAboveZone;

        return new Vector3(
            zone.x,
            safeY,
            zone.z
        );
    }

    // Selects the goal relevant to the current sub-task: target object, upper placement point, lower release point,
    // or final safe release position.
    private Vector3 GetActiveGoalPosition()
    {
        if (_released)
        {
            return GetSafeReleasePosition();
        }

        if (_lowering)
        {
            return GetLowerReleaseGoalPosition();
        }

        if (_attached)
        {
            return GetPlacementGoalPosition();
        }

        return targetObject != null ? targetObject.position : spawnCenter;
    }

    // Distance from the gripper to the target object, used by the Approach stage.
    private float GraspToTargetDist()
    {
        if (graspPoint == null || targetObject == null)
        {
            return 1f;
        }

        return Vector3.Distance(graspPoint.position, targetObject.position);
    }

    // Distance from the carried object to the upper placement goal, used by the Transport stage.
    private float TransportGoalDist()
    {
        if (targetObject == null)
        {
            return 1f;
        }

        return Vector3.Distance(targetObject.position, GetPlacementGoalPosition());
    }

    // Distance from the carried object to the lower release goal, used by the Placement stage.
    private float LowerGoalDist()
    {
        if (targetObject == null)
        {
            return 1f;
        }

        return Vector3.Distance(targetObject.position, GetLowerReleaseGoalPosition());
    }

    // Height of the carried object above the spawn plane, used for carry-height rewards and penalties.
    private float CarryHeightAboveSpawn()
    {
        if (targetObject == null)
        {
            return 0f;
        }

        return targetObject.position.y - spawnCenter.y;
    }

    // Height of the target above the placement zone base, used to decide when release is safe.
    private float HeightAboveZone()
    {
        if (targetObject == null)
        {
            return 1f;
        }

        return targetObject.position.y - GetZoneBaseY();
    }

    // Horizontal distance from the target to the centre of the placement zone.
    private float ReleaseXZDistance()
    {
        if (targetObject == null)
        {
            return 1f;
        }

        Vector3 zone = targetPlacement != null ? targetPlacement.position : GetLowerReleaseGoalPosition();

        if (safePlacementZone != null)
        {
            zone = safePlacementZone.transform.position;
        }

        Vector2 a = new Vector2(targetObject.position.x, targetObject.position.z);
        Vector2 b = new Vector2(zone.x, zone.z);

        return Vector2.Distance(a, b);
    }

    // Y position of the placement surface, with fallback values if placement references are missing.
    private float GetZoneBaseY()
    {
        if (targetPlacement != null)
        {
            return targetPlacement.position.y;
        }

        if (safePlacementZone != null)
        {
            return safePlacementZone.transform.position.y;
        }

        return spawnCenter.y;
    }

    // Editor-only debug visualisation of the approach radius, transport goal, release goal, placement tolerance, and spawn area.
    // These gizmos do not affect training.
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
        }

        Vector3 upperGoal = GetPlacementGoalPosition();
        Gizmos.color = new Color(1f, 0.3f, 1f, 0.75f);
        Gizmos.DrawWireSphere(upperGoal, placementRadius);

        Vector3 lowerGoal = GetLowerReleaseGoalPosition();
        Gizmos.color = new Color(0f, 1f, 0.4f, 0.85f);
        Gizmos.DrawWireSphere(lowerGoal, lowerReleaseRadius);

        if (targetPlacement != null)
        {
            Gizmos.color = new Color(0f, 1f, 1f, 0.8f);
            Gizmos.DrawWireSphere(targetPlacement.position, releaseStrictXZRadius);
        }

        Gizmos.color = new Color(1f, 1f, 0f, 0.3f);
        Gizmos.DrawWireCube(
            spawnCenter,
            new Vector3(spawnHalfSize * 2f, 0.02f, spawnHalfSize * 2f)
        );
    }
}