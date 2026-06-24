using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Reflection;

// Step 2 agent for the staged Pick-and-Place task, moving from the Approach stage into the Grasp and Transport stage described in the project brief.
// This phase extends the previous reaching policy by adding a scripted gripper closure and a simplified magnetic attachment once the end-effector is close enough to the object.
// The behaviour name, observation size, and action size are kept compatible with the previous stage.
public class STEP2_Grasp : Agent
{
    [Header("Robot Joints — ArticulationBody (shoulder -> wrist)")]
    // Six robot arm joints controlled by the RL policy.
    // Each joint receives one continuous action value.
    public ArticulationBody jointA1;
    public ArticulationBody jointA2;
    public ArticulationBody jointA3;
    public ArticulationBody jointA4;
    public ArticulationBody jointA5;
    public ArticulationBody jointA6;

    [Header("Joint angle limits (degrees)")]
    // Joint target angles are clamped to avoid unrealistic joint rotations.
    public float jointMinAngle = -120f;
    public float jointMaxAngle = 120f;

    [Header("End Point")]
    // The grasp point represents the end-effector used for measuring the distance to the target.
    public Transform graspPoint;

    [Header("Scene")]
    // The target object is the block to be reached and attached.
    // The robot base is used to avoid invalid target spawning too close to the arm.
    public Transform targetObject;
    public Transform robotBase;

    [Header("Gripper - optional")]
    // Optional gripper controller and finger joints.
    // The gripper is scripted in this stage so the RL policy only controls the arm joints.
    public MonoBehaviour gripperController;
    public ArticulationBody leftFingerJoint;
    public ArticulationBody rightFingerJoint;

    public float leftFingerOpenTarget = 0.01f;
    public float leftFingerClosedTarget = -0.01f;
    public float rightFingerOpenTarget = -0.01f;
    public float rightFingerClosedTarget = 0.01f;
    [Range(0.01f, 1f)] public float gripperCloseSpeed = 0.08f;

    [Header("Spawn Area — keep same as Step 1")]
    // Randomised target spawning supports generalisation and keeps Step 2 compatible with Step 1.
    public Vector3 spawnCenter = new Vector3(0.216f, 0.7f, -0.157f);
    public float spawnHalfSize = 0.1f;
    public float forbiddenRadius = 0.15f;
    public int spawnMaxTries = 20;

    [Header("Movement — balanced")]
    // Movement parameters scale and smooth the continuous actions before applying them to joint targets.
    public float jointSpeed = 0.8f;
    [Range(0.01f, 1f)] public float actionSmoothing = 0.18f;
    [Range(0.05f, 1f)] public float maxAbsAction = 0.45f;

    [Header("Anti-collapse reset / warmup")]
    // Warmup and reset options reduce unstable physical behaviour at the beginning of each episode.
    public int warmupDecisionSteps = 3;
    public bool hardResetJointPositions = true;
    public bool zeroJointVelocityOnReset = true;

    [Tooltip("Enable this only if the robot collapses immediately when Play starts. Normally, leave it unchecked.")]
    public bool strengthenJointDrives = false;
    public float minDriveStiffness = 10000f;
    public float minDriveDamping = 1000f;
    public float minDriveForceLimit = 1000f;

    [Header("Episode")]
    // Maximum episode length before timeout.
    public int maxEpisodeSteps = 4000;

    [Header("Step 2: grasp / attach")]
    // Attachment thresholds define when the gripper starts closing and when the object is magnetically attached.
    public float closeStartDistance = 0.095f;
    public float attachDistance = 0.06f;
    public int attachConfirmSteps = 2;
    public int holdAfterAttachSteps = 30;
    public Vector3 attachedLocalPosition = Vector3.zero;
    public Vector3 attachedLocalEuler = Vector3.zero;

    [Header("Step 2 physics simplification")]
    // Colliders are disabled during this simplified grasp stage to avoid unstable contact physics.
    public bool disableTargetCollidersDuringStep2 = true;

    [Header("Safety limits")]
    // Safety checks terminate episodes when the robot becomes unstable or leaves the useful workspace.
    public float minAllowedGraspPointY = 0.35f;
    public float maxWorkspaceDistanceFromSpawnCenter = 0.90f;
    public float maxAverageJointVelocity = 12f;
    public int unsafeVelocityStepLimit = 4;

    [Header("Reward weights")]
    // Reward shaping guides the policy from reaching to stable attachment.
    // The agent is rewarded for reducing distance, entering the close zone, becoming ready to attach, successfully attaching, and holding the object.
    public float improvementScale = 5f;
    public float distancePenaltyScale = 0.02f;
    public float timePenalty = -0.001f;
    public float stuckPenalty = -0.02f;
    public float closeZoneReward = 0.004f;
    public float attachReadyReward = 0.015f;
    public float attachBonus = 1.2f;
    public float holdReward = 0.018f;
    public float successReward = 2.5f;
    public float timeoutPenalty = -1.0f;
    public float unsafePenalty = -1.5f;
    public float jointVelocityPenaltyScale = 0.002f;
    public float actionChangePenaltyScale = 0.001f;

    [Header("Stuck detection")]
    // Stuck detection penalises the agent if distance to the target stops improving for too long.
    public float minImprovement = 0.001f;
    public int stuckStepLimit = 220;

    [Header("Debug")]
    // Debug gizmos visualise the target area and success thresholds in the Unity scene view.
    public bool drawDebugGizmos = true;

    // Initial joint targets used to reset the robot at the start of each episode.
    private float _restA1;
    private float _restA2;
    private float _restA3;
    private float _restA4;
    private float _restA5;
    private float _restA6;

    // Distance tracking for reward shaping and progress evaluation.
    private float _prevDist;
    private float _bestDist;
    private int _stuckSteps;
    private int _stepCount;
    private int _unsafeVelocitySteps;

    // Target physics and parenting state.
    private Rigidbody _targetRb;
    private Transform _targetOriginalParent;
    private Quaternion _targetOriginalRotation;

    // Grasp and attachment state.
    private bool _attached;
    private int _attachZoneSteps;
    private int _attachedHoldSteps;
    private float _gripperClose01;

    // Action smoothing buffers.
    private readonly float[] _smoothedActions = new float[6];
    private readonly float[] _lastAppliedActions = new float[6];

    // Position normalisation scale for vector observations.
    private const float OBS_NORM = 0.5f;

    public override void Initialize()
    {
        if (strengthenJointDrives)
        {
            StrengthenAllDrives();
        }

        // Use the current scene pose as the reset pose.
        // This avoids a mismatch between xDrive targets and the actual joint positions.
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

        PrepareTargetPhysicsBeforeAttach();
    }

    public override void OnEpisodeBegin()
    {
        // Reset the arm, gripper, target, and internal training state for a new episode.
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
        _attachedHoldSteps = 0;
        _gripperClose01 = 0f;
        _unsafeVelocitySteps = 0;

        OpenGripperImmediately();
        ResetTargetObject();

        _stepCount = 0;
        _stuckSteps = 0;
        _prevDist = GraspToTargetDist();
        _bestDist = _prevDist;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Observation space:
        // 6 normalised joint angles + end-effector position + target position + vector to target + scalar progress values.
        // The total observation size remains 20 for compatibility with the previous reaching model.
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

        Vector3 tgRel = targetObject != null
            ? (targetObject.position - spawnCenter) / OBS_NORM
            : Vector3.zero;
        sensor.AddObservation(ClampVec(tgRel));

        Vector3 toTarget = targetObject != null && graspPoint != null
            ? targetObject.position - graspPoint.position
            : Vector3.zero;
        sensor.AddObservation(ClampVec(toTarget));

        sensor.AddObservation(Mathf.Clamp(toTarget.magnitude, 0f, 5f));
        sensor.AddObservation(Mathf.Clamp(graspPoint != null ? graspPoint.position.y - spawnCenter.y : 0f, -5f, 5f));
        sensor.AddObservation((float)_stepCount / Mathf.Max(1, maxEpisodeSteps));
        sensor.AddObservation(Mathf.Clamp01((float)_stuckSteps / Mathf.Max(1, stuckStepLimit)));
        sensor.AddObservation(Mathf.Clamp01(_bestDist));
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        _stepCount++;

        if (_stepCount <= warmupDecisionSteps)
        {
            // During the warmup period, hold the reset pose so the robot starts stably.
            HoldRestPoseDuringWarmup();
            UpdateScriptedGripper();
            AddReward(timePenalty);
            _prevDist = GraspToTargetDist();
            return;
        }

        // Apply the policy actions to the six arm joints and update the scripted gripper.
        ApplySafeJointActions(actions);
        UpdateScriptedGripper();

        float currDist = GraspToTargetDist();

        // Main shaping reward: positive when the end-effector moves closer to the target.
        AddReward((_prevDist - currDist) * improvementScale);
        AddReward(-currDist * distancePenaltyScale * Time.fixedDeltaTime);
        AddReward(timePenalty);

        if (!_attached)
        {
            // Before attachment, the policy is guided toward the target and rewarded for entering the grasp zone.
            if (currDist < closeStartDistance)
            {
                AddReward(closeZoneReward);
            }

            if (currDist < attachDistance)
            {
                _attachZoneSteps++;
                AddReward(attachReadyReward);
            }
            else
            {
                _attachZoneSteps = 0;
            }

            // The object is attached only after the gripper has closed enough and the end-effector remains close.
            if (_attachZoneSteps >= attachConfirmSteps && _gripperClose01 >= 0.55f)
            {
                AttachTargetToGraspPoint();
                AddReward(attachBonus);
            }
        }
        else
        {
            // After attachment, keep the object locked to the grasp point and reward stable holding.
            KeepTargetLockedToGraspPoint();
            _attachedHoldSteps++;
            AddReward(holdReward);

            float lockError = targetObject != null && graspPoint != null
                ? Vector3.Distance(targetObject.position, graspPoint.position)
                : 1f;

            AddReward(-lockError * 0.03f);

            if (_attachedHoldSteps >= holdAfterAttachSteps)
            {
                AddReward(successReward);
                EndEpisode();
                return;
            }
        }

        AddStuckPenalty(currDist);
        AddSmoothnessAndVelocityPenalty(actions);

        if (currDist < _bestDist)
        {
            _bestDist = currDist;
        }

        _prevDist = currDist;

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

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // Manual keyboard control for debugging the six joint actions.
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

    private void ApplySafeJointActions(ActionBuffers actions)
    {
        float dist = GraspToTargetDist();

        // The policy outputs one continuous value per joint.
        // Actions are smoothed and scaled to reduce shaking near the object.
        float nearScale = 1f;

        if (dist < closeStartDistance)
        {
            nearScale = 0.85f;
        }

        if (dist < attachDistance * 1.35f)
        {
            nearScale = 0.65f;
        }

        AddDeltaTarget(jointA1, SmoothAction(actions.ContinuousActions[0], 0) * jointSpeed * 1.00f * nearScale);
        AddDeltaTarget(jointA2, SmoothAction(actions.ContinuousActions[1], 1) * jointSpeed * 0.95f * nearScale);
        AddDeltaTarget(jointA3, SmoothAction(actions.ContinuousActions[2], 2) * jointSpeed * 0.90f * nearScale);
        AddDeltaTarget(jointA4, SmoothAction(actions.ContinuousActions[3], 3) * jointSpeed * 0.70f * nearScale);
        AddDeltaTarget(jointA5, SmoothAction(actions.ContinuousActions[4], 4) * jointSpeed * 0.60f * nearScale);
        AddDeltaTarget(jointA6, SmoothAction(actions.ContinuousActions[5], 5) * jointSpeed * 0.50f * nearScale);
    }

    private float SmoothAction(float rawAction, int index)
    {
        // Clamp and smooth actions before applying them to joint drive targets.
        float clamped = Mathf.Clamp(rawAction, -maxAbsAction, maxAbsAction);
        _smoothedActions[index] = Mathf.Lerp(_smoothedActions[index], clamped, actionSmoothing);
        return _smoothedActions[index];
    }

    private void UpdateScriptedGripper()
    {
        float dist = GraspToTargetDist();

        float targetClose = dist < closeStartDistance || _attached ? 1f : 0f;

        // The gripper closes gradually when the end-effector enters the close zone.
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

    private void OpenGripperImmediately()
    {
        // Reset both finger joints to the open pose.
        _gripperClose01 = 0f;
        SetRawTarget(leftFingerJoint, leftFingerOpenTarget);
        SetRawTarget(rightFingerJoint, rightFingerOpenTarget);
        DriveOptionalGripperController(0f);
    }

    private void DriveOptionalGripperController(float close01)
    {
        // Call optional gripper control methods if the scene uses an external gripper controller.
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

    private void CallOptionalVoidMethod(MonoBehaviour target, string methodName)
    {
        // Reflection allows compatibility with different gripper controller method names.
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

    private void CallOptionalFloatMethod(MonoBehaviour target, string methodName, float value)
    {
        // Reflection is used so this agent can work with several controller implementations.
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

    private void ResetTargetObject()
    {
        // Restore the target to its original parent, randomise its position, and disable unstable physics.
        if (targetObject == null)
        {
            return;
        }

        targetObject.SetParent(_targetOriginalParent, true);
        targetObject.position = GetValidSpawnPosition();
        targetObject.rotation = _targetOriginalRotation;

        PrepareTargetPhysicsBeforeAttach();
    }

    private void PrepareTargetPhysicsBeforeAttach()
    {
        // The object is treated as kinematic before attachment.
        // This focuses the learning task on reaching and grasp triggering rather than contact physics.
        if (_targetRb == null && targetObject != null)
        {
            _targetRb = targetObject.GetComponent<Rigidbody>();
        }

        if (_targetRb != null)
        {
            if (!_targetRb.isKinematic)
            {
                _targetRb.linearVelocity = Vector3.zero;
                _targetRb.angularVelocity = Vector3.zero;
            }

            _targetRb.useGravity = false;
            _targetRb.detectCollisions = false;
            _targetRb.isKinematic = true;
        }

        if (disableTargetCollidersDuringStep2 && targetObject != null)
        {
            Collider[] cols = targetObject.GetComponentsInChildren<Collider>();

            foreach (Collider c in cols)
            {
                c.enabled = false;
            }
        }
    }

    private void AttachTargetToGraspPoint()
    {
        // Simplified magnetic grasp:
        // once the end-effector and gripper satisfy the attach condition, the object is parented to the grasp point instead of relying on unstable physical gripping.
        if (targetObject == null || graspPoint == null)
        {
            return;
        }

        _attached = true;
        _attachedHoldSteps = 0;

        if (_targetRb != null)
        {
            if (!_targetRb.isKinematic)
            {
                _targetRb.linearVelocity = Vector3.zero;
                _targetRb.angularVelocity = Vector3.zero;
            }

            _targetRb.useGravity = false;
            _targetRb.detectCollisions = false;
            _targetRb.isKinematic = true;
        }

        if (disableTargetCollidersDuringStep2)
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

    private void KeepTargetLockedToGraspPoint()
    {
        // Maintain the simplified grasp by keeping the object fixed to the end-effector.
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

    private void AddStuckPenalty(float currDist)
    {
        // Penalise repeated steps that do not reduce the distance to the target.
        if (_prevDist - currDist > minImprovement)
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

    private void AddSmoothnessAndVelocityPenalty(ActionBuffers actions)
    {
        // Penalise abrupt action changes and excessive joint velocity to improve motion stability.
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

    private bool IsUnsafeState()
    {
        // End the episode if the arm becomes unstable, leaves the workspace, or moves too fast.
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

    private float AverageJointVelocityAbs()
    {
        // Compute the average absolute joint velocity across the six controlled joints.
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

    private void AddJointVelocity(ArticulationBody ab, ref float sum, ref int count)
    {
        if (ab == null || ab.jointVelocity.dofCount <= 0)
        {
            return;
        }

        sum += Mathf.Abs(ab.jointVelocity[0]);
        count++;
    }

    private void HoldRestPoseDuringWarmup()
    {
        // Hold the initial pose for a few steps to avoid reset instability.
        SetTarget(jointA1, _restA1);
        SetTarget(jointA2, _restA2);
        SetTarget(jointA3, _restA3);
        SetTarget(jointA4, _restA4);
        SetTarget(jointA5, _restA5);
        SetTarget(jointA6, _restA6);

        ZeroAllJointVelocities();
    }

    private void StrengthenAllDrives()
    {
        // Optionally increase drive strength on all joints for a more stable articulation setup.
        StrengthenDrive(jointA1);
        StrengthenDrive(jointA2);
        StrengthenDrive(jointA3);
        StrengthenDrive(jointA4);
        StrengthenDrive(jointA5);
        StrengthenDrive(jointA6);
    }

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

    private void ResetJointToSafePose(ArticulationBody ab, float targetDeg)
    {
        // Reset both the drive target and optionally the actual joint position.
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

    private void ZeroAllJointVelocities()
    {
        // Clear joint velocities to prevent motion from carrying over between episodes.
        ZeroJointVelocity(jointA1);
        ZeroJointVelocity(jointA2);
        ZeroJointVelocity(jointA3);
        ZeroJointVelocity(jointA4);
        ZeroJointVelocity(jointA5);
        ZeroJointVelocity(jointA6);
    }

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

    private float GetCurrentAngleOrTarget(ArticulationBody ab)
    {
        // Read the current joint angle if available; otherwise use the xDrive target.
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

    private float GetTarget(ArticulationBody ab)
    {
        if (ab == null)
        {
            return 0f;
        }

        return ab.xDrive.target;
    }

    private void SetTarget(ArticulationBody ab, float target)
    {
        // Set the joint drive target within the configured angle limits.
        if (ab == null)
        {
            return;
        }

        var drive = ab.xDrive;
        drive.target = Mathf.Clamp(target, jointMinAngle, jointMaxAngle);
        ab.xDrive = drive;
    }

    private void SetRawTarget(ArticulationBody ab, float target)
    {
        // Set a raw drive target for gripper finger joints.
        if (ab == null)
        {
            return;
        }

        var drive = ab.xDrive;
        drive.target = target;
        ab.xDrive = drive;
    }

    private void AddDeltaTarget(ArticulationBody ab, float delta)
    {
        // Apply incremental joint target updates generated by the policy.
        if (ab == null)
        {
            return;
        }

        var drive = ab.xDrive;
        drive.target = Mathf.Clamp(drive.target + delta, jointMinAngle, jointMaxAngle);
        ab.xDrive = drive;
    }

    private float NormAngle(ArticulationBody ab)
    {
        // Convert the joint position from radians to degrees and normalise it for the observation vector.
        if (ab == null || ab.jointPosition.dofCount <= 0)
        {
            return 0f;
        }

        float angle = ab.jointPosition[0] * Mathf.Rad2Deg;
        return Mathf.Clamp(angle / 180f, -1f, 1f);
    }

    private Vector3 ClampVec(Vector3 v)
    {
        // Clamp vector observations to reduce extreme input values.
        return new Vector3(
            Mathf.Clamp(v.x, -5f, 5f),
            Mathf.Clamp(v.y, -5f, 5f),
            Mathf.Clamp(v.z, -5f, 5f)
        );
    }

    private Vector3 GetValidSpawnPosition()
    {
        // Sample a valid target position while avoiding locations too close to the robot base.
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

    private float GraspToTargetDist()
    {
        // Main task metric used for rewards, progress tracking, and attachment conditions.
        if (graspPoint == null || targetObject == null)
        {
            return 1f;
        }

        return Vector3.Distance(graspPoint.position, targetObject.position);
    }

    private void OnDrawGizmos()
    {
        // Draw debug visuals for the attach distance, close-start distance, and random spawn area.
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