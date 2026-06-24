using UnityEngine; 
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

// This agent implements the Approach stage of the Pick-and-Place pipeline described in the project brief: reaching.
// The goal is to train the robotic arm to move its end-effector, represented by graspPoint, toward a randomly spawned target object.
public class NiryoReachAgent : Agent
{
    [Header("Robot Joints — ArticulationBody (shoulder -> wrist)")]
    // Six controllable ArticulationBody joints of the robotic arm.
    // These joints form the continuous action space of the RL agent.
    public ArticulationBody jointA1;
    public ArticulationBody jointA2;
    public ArticulationBody jointA3;
    public ArticulationBody jointA4;
    public ArticulationBody jointA5;
    public ArticulationBody jointA6;

    [Header("Joint angle limits (degrees)")]
    // Joint targets are clamped within this range to prevent unrealistic or unstable motions.
    public float jointMinAngle = -120f;
    public float jointMaxAngle =  120f;

    [Header("End Point")]
    // The graspPoint represents the end-effector position used to measure distance to the target.
    public Transform graspPoint;

    [Header("Scene")]
    // targetObject is the object that the end-effector should approach.
    // robotBase is used to avoid spawning the target too close to the robot base.
    public Transform targetObject;
    public Transform robotBase;

    [Header("Spawn Area")]
    // The target is randomly spawned inside a square region around spawnCenter.
    // This supports task randomisation and improves generalisation.
    public Vector3 spawnCenter    = new Vector3(0.216f, 0.7f, -0.157f);
    public float   spawnHalfSize  = 0.08f;
    public float   forbiddenRadius = 0.15f;
    public int     spawnMaxTries  = 20;

    [Header("Movement")]
    // This controls how strongly each continuous action changes the joint target angle.
    public float jointSpeed = 0.5f;

    [Header("Episode")]
    // Maximum number of decision steps before the episode is treated as a timeout.
    public int maxEpisodeSteps = 3000;

    [Header("Success")]
    // If the end-effector is closer than this threshold to the target, the reaching task succeeds.
    public float touchThreshold = 0.06f;

    [Header("Reward weights")]
    // Reward design:
    // improvementScale rewards distance reduction between the graspPoint and target.
    // distancePenaltyScale encourages the agent to stay close to the target.
    // timePenalty encourages efficient reaching.
    // stuckPenalty discourages repeated steps without meaningful progress.
    // successReward is given when the end-effector reaches the target.
    // timeoutPenalty is applied if the agent fails within the episode limit.
    public float improvementScale     = 5f;
    public float distancePenaltyScale = 0.02f;
    public float timePenalty          = -0.001f;
    public float stuckPenalty         = -0.02f;
    public float successReward        = 2.0f;
    public float timeoutPenalty       = -1.0f;

    [Header("Stuck detection")]
    // These variables identify whether the agent is failing to make progress.
    // If the distance improvement is smaller than minImprovement for too long, the agent receives an additional penalty.
    public float minImprovement = 0.001f;
    public int   stuckStepLimit = 80;

    // Initial rest targets of the six joints.
    // These are stored at the beginning and restored at the start of each episode.
    private float _restA1, _restA2, _restA3, _restA4, _restA5, _restA6;

    // Distance tracking variables used for progress reward and stuck detection.
    private float _prevDist;
    private float _bestDist;

    // Episode counters.
    private int   _stuckSteps;
    private int   _stepCount;

    // Normalisation factor for position observations.
    // It keeps input values within a smaller numerical range for more stable learning.
    private const float OBS_NORM = 0.5f;

    public override void Initialize()
    {
        // Store the initial joint drive targets as the reset pose.
        _restA1 = GetTarget(jointA1);
        _restA2 = GetTarget(jointA2);
        _restA3 = GetTarget(jointA3);
        _restA4 = GetTarget(jointA4);
        _restA5 = GetTarget(jointA5);
        _restA6 = GetTarget(jointA6);

        // The target is made kinematic because this reaching stage only needs a static object.
        // Physical pushing or grasping is not required in this phase.
        var rb = targetObject ? targetObject.GetComponent<Rigidbody>() : null;
        if (rb != null) { rb.isKinematic = true; rb.useGravity = false; }
    }

    public override void OnEpisodeBegin()
    {
        // Reset the robot to the original joint configuration at the beginning of each episode.
        SetTarget(jointA1, _restA1);
        SetTarget(jointA2, _restA2);
        SetTarget(jointA3, _restA3);
        SetTarget(jointA4, _restA4);
        SetTarget(jointA5, _restA5);
        SetTarget(jointA6, _restA6);

        // Randomise the target position for each episode.
        // This prevents the policy from overfitting to one fixed target location.
        targetObject.position = GetValidSpawnPosition();

        // Reset episode state variables.
        _stepCount  = 0;
        _stuckSteps = 0;
        _prevDist   = GraspToTargetDist();
        _bestDist   = _prevDist;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Observation group 1:
        // Normalised joint angles of the six robotic arm joints.
        // These describe the current robot state.
        sensor.AddObservation(NormAngle(jointA1));
        sensor.AddObservation(NormAngle(jointA2));
        sensor.AddObservation(NormAngle(jointA3));
        sensor.AddObservation(NormAngle(jointA4));
        sensor.AddObservation(NormAngle(jointA5));
        sensor.AddObservation(NormAngle(jointA6));

        // Observation group 2:
        // End-effector position relative to the spawn center.
        // This tells the agent where the gripper is in the workspace.
        Vector3 gpRel = graspPoint
            ? (graspPoint.position - spawnCenter) / OBS_NORM
            : Vector3.zero;
        sensor.AddObservation(ClampVec(gpRel));

        // Observation group 3:
        // Target position relative to the spawn center.
        // This gives the agent the target location in a normalised coordinate frame.
        Vector3 tgRel = targetObject
            ? (targetObject.position - spawnCenter) / OBS_NORM
            : Vector3.zero;
        sensor.AddObservation(ClampVec(tgRel));

        // Observation group 4:
        // Direction vector from the end-effector to the target.
        // This is the most direct task-related observation for reaching.
        Vector3 toTarget = targetObject && graspPoint
            ? targetObject.position - graspPoint.position
            : Vector3.zero;
        sensor.AddObservation(ClampVec(toTarget));

        // Additional scalar observations:
        // distance to target, end-effector height, episode progress, stuck ratio, and best distance.
        sensor.AddObservation(Mathf.Clamp(toTarget.magnitude, 0f, 5f));
        sensor.AddObservation(Mathf.Clamp(
            graspPoint ? graspPoint.position.y - spawnCenter.y : 0f, -5f, 5f));
        sensor.AddObservation((float)_stepCount / maxEpisodeSteps);
        sensor.AddObservation(Mathf.Clamp01((float)_stuckSteps / stuckStepLimit));
        sensor.AddObservation(Mathf.Clamp01(_bestDist));
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        _stepCount++;

        // Continuous action space:
        // The policy outputs six continuous values, one for each joint.
        // Each value is scaled by jointSpeed and applied as a delta to the joint target angle.
        AddDeltaTarget(jointA1, actions.ContinuousActions[0] * jointSpeed);
        AddDeltaTarget(jointA2, actions.ContinuousActions[1] * jointSpeed);
        AddDeltaTarget(jointA3, actions.ContinuousActions[2] * jointSpeed);
        AddDeltaTarget(jointA4, actions.ContinuousActions[3] * jointSpeed);
        AddDeltaTarget(jointA5, actions.ContinuousActions[4] * jointSpeed);
        AddDeltaTarget(jointA6, actions.ContinuousActions[5] * jointSpeed);

        // Current distance between the end-effector and the target.
        float currDist = GraspToTargetDist();

        // Reward component 1:
        // Positive reward when the agent reduces the distance to the target.
        AddReward((_prevDist - currDist) * improvementScale);

        // Reward component 2:
        // Small continuous penalty proportional to the current distance.
        // This encourages the arm to remain close to the target.
        AddReward(-currDist * distancePenaltyScale * Time.fixedDeltaTime);

        // Reward component 3:
        // Time penalty encourages the policy to complete the reaching task efficiently.
        AddReward(timePenalty);

        // Stuck detection:
        // If the agent is not improving enough, count stuck steps and penalise prolonged stagnation.
        if (_prevDist - currDist > minImprovement)
            _stuckSteps = 0;
        else
        {
            _stuckSteps++;
            if (_stuckSteps > stuckStepLimit)
                AddReward(stuckPenalty);
        }

        // Track the best distance reached during the current episode.
        if (currDist < _bestDist) _bestDist = currDist;
        _prevDist = currDist;

        // Success condition:
        // The episode ends when the end-effector reaches the target within the touch threshold.
        if (currDist < touchThreshold)
        {
            AddReward(successReward);
            EndEpisode();
            return;
        }

        // Timeout condition:
        // If the maximum episode length is reached, the attempt is considered unsuccessful.
        if (_stepCount >= maxEpisodeSteps)
        {
            AddReward(timeoutPenalty);
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // Manual keyboard control for debugging.
        // This allows testing each joint before or during RL training.
        var ca = actionsOut.ContinuousActions;
        for (int i = 0; i < 6; i++) ca[i] = 0f;

        if (Input.GetKey(KeyCode.Alpha1)) ca[0] =  1f;
        if (Input.GetKey(KeyCode.Q))      ca[0] = -1f;
        if (Input.GetKey(KeyCode.Alpha2)) ca[1] =  1f;
        if (Input.GetKey(KeyCode.W))      ca[1] = -1f;
        if (Input.GetKey(KeyCode.Alpha3)) ca[2] =  1f;
        if (Input.GetKey(KeyCode.E))      ca[2] = -1f;
        if (Input.GetKey(KeyCode.Alpha4)) ca[3] =  1f;
        if (Input.GetKey(KeyCode.R))      ca[3] = -1f;
        if (Input.GetKey(KeyCode.Alpha5)) ca[4] =  1f;
        if (Input.GetKey(KeyCode.T))      ca[4] = -1f;
        if (Input.GetKey(KeyCode.Alpha6)) ca[5] =  1f;
        if (Input.GetKey(KeyCode.Y))      ca[5] = -1f;
    }

    private float GetTarget(ArticulationBody ab)
    {
        // Return the current xDrive target angle of a joint.
        // If the joint reference is missing, return zero to avoid null reference errors.
        if (ab == null) return 0f;
        return ab.xDrive.target;
    }

    private void SetTarget(ArticulationBody ab, float target)
    {
        // Set a joint target while clamping it within the allowed angular range.
        // This helps keep the robot motion physically reasonable.
        if (ab == null) return;
        var drive = ab.xDrive;
        drive.target = Mathf.Clamp(target, jointMinAngle, jointMaxAngle);
        ab.xDrive = drive;
    }

    private void AddDeltaTarget(ArticulationBody ab, float delta)
    {
        // Apply an incremental change to the joint target.
        // RL actions are therefore interpreted as velocity-like joint target updates.
        if (ab == null) return;
        var drive = ab.xDrive;
        drive.target = Mathf.Clamp(drive.target + delta, jointMinAngle, jointMaxAngle);
        ab.xDrive = drive;
    }

    private float NormAngle(ArticulationBody ab)
    {
        // Convert the current joint position from radians to degrees, then normalise it to roughly the range [-1, 1].
        if (ab == null) return 0f;
        float angle = ab.jointPosition.dofCount > 0
            ? ab.jointPosition[0] * Mathf.Rad2Deg
            : 0f;
        return Mathf.Clamp(angle / 180f, -1f, 1f);
    }

    private Vector3 ClampVec(Vector3 v)
    {
        // Clamp vector observations to avoid very large input values, which can make neural network training less stable.
        return new Vector3(
            Mathf.Clamp(v.x, -5f, 5f),
            Mathf.Clamp(v.y, -5f, 5f),
            Mathf.Clamp(v.z, -5f, 5f));
    }

    private Vector3 GetValidSpawnPosition()
    {
        // Generate a random target position inside the spawn area.
        // Positions too close to the robot base are rejected to avoid invalid or unfair starts.
        Vector3 robotPos = robotBase ? robotBase.position : Vector3.zero;

        for (int i = 0; i < spawnMaxTries; i++)
        {
            float rx = Random.Range(-spawnHalfSize, spawnHalfSize);
            float rz = Random.Range(-spawnHalfSize, spawnHalfSize);
            Vector3 candidate = new Vector3(
                spawnCenter.x + rx,
                spawnCenter.y,
                spawnCenter.z + rz);

            float dxz = Vector2.Distance(
                new Vector2(candidate.x, candidate.z),
                new Vector2(robotPos.x, robotPos.z));

            if (dxz >= forbiddenRadius)
                return candidate;
        }

        // Fallback position if no valid random sample is found.
        return spawnCenter + new Vector3(spawnHalfSize * 0.5f, 0f, 0f);
    }

    private float GraspToTargetDist()
    {
        // Main task metric: Euclidean distance between the end-effector and the target.
        // This value is used for progress rewards, success checking, and evaluation.
        if (graspPoint == null || targetObject == null) return 1f;
        return Vector3.Distance(graspPoint.position, targetObject.position);
    }

    private void OnDrawGizmos()
    {
        // Visual debugging:
        // Red line shows the current distance between the gripper and target.
        // Green sphere shows the success threshold around the target.
        if (graspPoint != null && targetObject != null)
        {
            Gizmos.color = Color.red;
            Gizmos.DrawLine(graspPoint.position, targetObject.position);
            Gizmos.color = new Color(0f, 1f, 0f, 0.2f);
            Gizmos.DrawSphere(targetObject.position, touchThreshold);
        }

        // Yellow wire cube shows the random spawn region.
        Gizmos.color = new Color(1f, 1f, 0f, 0.3f);
        Gizmos.DrawWireCube(spawnCenter,
            new Vector3(spawnHalfSize * 2f, 0.02f, spawnHalfSize * 2f));

        // Red wire sphere shows the forbidden spawn radius around the robot base.
        if (robotBase != null)
        {
            Gizmos.color = new Color(1f, 0f, 0f, 0.2f);
            Gizmos.DrawWireSphere(
                new Vector3(robotBase.position.x, spawnCenter.y, robotBase.position.z),
                forbiddenRadius);
        }
    }
}