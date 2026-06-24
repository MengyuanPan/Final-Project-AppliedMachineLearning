using UnityEngine;

// Simple gripper helper for the Niryo One pick-and-place setup.
// The main RL agents control the arm joints, while this script provides a separate open/close interface for the gripper.
// It writes directly to the ArticulationBody xDrive targets of the left and right gripper fingers.
// The open and close target values match the values used by the Unity-Robotics-Hub TrajectoryPlanner setup,
// so the gripper movement stays consistent with the original robot example.
public class SimpleGripperController : MonoBehaviour
{
    // ArticulationBody references for the two physical gripper fingers.
    // These are assigned in the Unity Inspector by dragging in the left and right gripper objects.
    [Header("Gripper ArticulationBodies")]
    [Tooltip("Drag left_gripper GameObject here.")]
    public ArticulationBody leftGripper;

    [Tooltip("Drag right_gripper GameObject here.")]
    public ArticulationBody rightGripper;

    // xDrive target values for the open and closed finger poses.
    // The left and right fingers use opposite signs because they move in mirrored directions.
    [Header("Drive targets (from official TrajectoryPlanner.cs)")]
    public float leftOpenTarget    =  0.01f;
    public float leftClosedTarget  = -0.01f;
    public float rightOpenTarget   = -0.01f;
    public float rightClosedTarget =  0.01f;

    // Optional keyboard controls for testing the gripper in Play mode without running the RL agent.
    [Header("Keyboard test (Play mode)")]
    public bool enableKeyboardTest = true;
    public KeyCode closeKey = KeyCode.C;
    public KeyCode openKey  = KeyCode.O;

    // Internal state used by other scripts to check whether the gripper is currently treated as closed.
    private bool _isClosed = false;

    // Start with the gripper open so each scene run begins from a predictable grasp state.
    private void Start()
    {
        OpenGripper();
    }

    // Keyboard test path for manually checking the gripper open/close targets during Play mode.
    private void Update()
    {
        if (!enableKeyboardTest) return;
        if (Input.GetKeyDown(closeKey)) CloseGripper();
        if (Input.GetKeyDown(openKey))  OpenGripper();
    }

    // Opens both gripper fingers by applying the configured open targets.
    public void OpenGripper()
    {
        _isClosed = false;
        SetTarget(leftGripper,  leftOpenTarget);
        SetTarget(rightGripper, rightOpenTarget);
    }

    // Closes both gripper fingers by applying the configured closed targets.
    public void CloseGripper()
    {
        _isClosed = true;
        SetTarget(leftGripper,  leftClosedTarget);
        SetTarget(rightGripper, rightClosedTarget);
    }

    // Returns the scripted gripper state for other scripts that need to check grasp status.
    public bool IsClosed() { return _isClosed; }

    // Writes a target value into one gripper finger's xDrive.
    // Missing references are ignored so the scene does not crash while the Inspector is being set up.
    private void SetTarget(ArticulationBody ab, float target)
    {
        if (ab == null) return;
        var drive = ab.xDrive;
        drive.target = target;
        ab.xDrive = drive;
    }
}