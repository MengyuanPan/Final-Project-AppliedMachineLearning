using UnityEngine;

// Safe placement zone helper for the pick-and-place environment.
// This script builds a visible square frame, adds an optional trigger area, and randomises the placement goal
// so the agent has to place the object in different valid positions instead of always using one fixed target.
// The random position is checked against the robot base and the target object to keep the placement zone reachable,
// clear of the robot, and separate from the object spawn position.

public class Safeplacement : MonoBehaviour
{
    // Scene references used to define where the placement zone can appear.
    // referenceCenter is normally a stable point in the workspace, robotBase is used for distance checks,
    // and targetObject is used to avoid placing the goal too close to the spawned block.
    [Header("References")]
    public Transform referenceCenter;
    public Transform robotBase;
    public Transform targetObject;

    // Randomisation area for the placement zone.
    // If useReferenceCenter is enabled, the area is placed relative to referenceCenter; otherwise the offset is used as a world position.
    [Header("Placement random area")]
    public bool useReferenceCenter = true;
    public Vector3 areaCenterOffset = new Vector3(0.32f, 0f, 0.12f);
    public float fixedY = 0.70f;

    public float areaHalfSizeX = 0.16f;
    public float areaHalfSizeZ = 0.14f;

    // Safety distance settings for valid placement-zone samples.
    // The zone should not be too close to the robot base, too far for the arm to reach, or too close to the target object.
    [Header("Safety distance")]
    public float minDistanceFromRobot = 0.32f;
    public float maxDistanceFromRobot = 0.75f;
    public float minDistanceFromTarget = 0.22f;
    public int maxRandomTries = 80;

    // Visual frame settings.
    // The four bars make the placement area visible in the scene, which helps with debugging and recording results.
    [Header("Frame visual")]
    public float frameSize = 0.20f;
    public float barThickness = 0.015f;
    public float barHeight = 0.015f;
    public Color frameColor = new Color(0f, 1f, 1f, 1f);

    // Optional trigger box used as a simple placement-area detector.
    // Its size follows the frame so the visible target area and trigger area match.
    [Header("Trigger box")]
    public bool addTriggerCollider = true;
    public float triggerHeight = 0.08f;

    // Draw gizmos in the Scene view to show the randomisation area and safety-distance limits.
    [Header("Debug")]
    public bool drawGizmos = true;

    // Runtime material shared by the frame bars so their colour can be controlled from this script.
    private Material _runtimeMaterial;

    // Build the frame and trigger when the object is created during play mode.
    private void Awake()
    {
        BuildFrame();
        SetupTriggerCollider();
    }

    // Rebuild the helper objects when the component is reset in the Unity Editor.
    private void Reset()
    {
        BuildFrame();
        SetupTriggerCollider();
    }

    // Creates or updates the four frame bars that mark the placement zone.
    [ContextMenu("Build Frame")]
    public void BuildFrame()
    {
        CreateOrUpdateBar(
            "FrontBar",
            new Vector3(0f, 0f, frameSize * 0.5f),
            new Vector3(frameSize, barHeight, barThickness)
        );

        CreateOrUpdateBar(
            "BackBar",
            new Vector3(0f, 0f, -frameSize * 0.5f),
            new Vector3(frameSize, barHeight, barThickness)
        );

        CreateOrUpdateBar(
            "LeftBar",
            new Vector3(-frameSize * 0.5f, 0f, 0f),
            new Vector3(barThickness, barHeight, frameSize)
        );

        CreateOrUpdateBar(
            "RightBar",
            new Vector3(frameSize * 0.5f, 0f, 0f),
            new Vector3(barThickness, barHeight, frameSize)
        );
    }

    // Editor context-menu shortcut for testing one random placement without starting training.
    [ContextMenu("Randomize Once")]
    public void RandomizeOnce()
    {
        RandomizePosition();
    }

    // Returns a goal point above the centre of the placement zone.
    // The agent can use this as an upper transport goal before lowering the object for release.
    public Vector3 GetGoalPosition(float aboveY)
    {
        return transform.position + Vector3.up * aboveY;
    }

    // Samples a valid placement-zone position inside the configured random area.
    // Each candidate is accepted only if it is far enough from the robot, still reachable by the arm,
    // and separated from the current target object position.
    public void RandomizePosition()
    {
        Vector3 areaCenter = GetAreaCenterWorld();
        Vector3 robotPos = robotBase != null ? robotBase.position : Vector3.zero;
        Vector3 targetPos = targetObject != null ? targetObject.position : areaCenter;

        for (int i = 0; i < maxRandomTries; i++)
        {
            Vector3 candidate = new Vector3(
                areaCenter.x + Random.Range(-areaHalfSizeX, areaHalfSizeX),
                fixedY,
                areaCenter.z + Random.Range(-areaHalfSizeZ, areaHalfSizeZ)
            );

            float distRobot = XZDistance(candidate, robotPos);
            float distTarget = XZDistance(candidate, targetPos);

            bool safeFromRobot = distRobot >= minDistanceFromRobot;
            bool reachable = distRobot <= maxDistanceFromRobot;
            bool separatedFromTarget = distTarget >= minDistanceFromTarget;

            if (safeFromRobot && reachable && separatedFromTarget)
            {
                transform.position = candidate;
                return;
            }
        }

        transform.position = new Vector3(areaCenter.x, fixedY, areaCenter.z);
    }

    // Checks whether a target is horizontally inside the placement frame.
    // Only the XZ plane is checked here because placement height is handled separately by the agent.
    public bool IsTargetInsideXZ(Transform target, float tolerance = 0.02f)
    {
        if (target == null)
        {
            return false;
        }

        float half = frameSize * 0.5f + tolerance;
        Vector3 local = transform.InverseTransformPoint(target.position);

        return Mathf.Abs(local.x) <= half && Mathf.Abs(local.z) <= half;
    }

    // Calculates the world-space centre of the randomisation area.
    // Using a reference centre keeps the placement area aligned with the workspace even if the scene is moved.
    private Vector3 GetAreaCenterWorld()
    {
        if (useReferenceCenter && referenceCenter != null)
        {
            return new Vector3(
                referenceCenter.position.x + areaCenterOffset.x,
                fixedY,
                referenceCenter.position.z + areaCenterOffset.z
            );
        }

        return new Vector3(
            areaCenterOffset.x,
            fixedY,
            areaCenterOffset.z
        );
    }

    // Creates one visual frame bar if it does not exist, or updates it if it already exists.
    // Primitive cube colliders are removed so the visual frame does not interfere with robot or object movement.
    private void CreateOrUpdateBar(string barName, Vector3 localPos, Vector3 localScale)
    {
        Transform bar = transform.Find(barName);

        if (bar == null)
        {
            GameObject go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = barName;
            go.transform.SetParent(transform, false);

            Collider col = go.GetComponent<Collider>();
            if (col != null)
            {
                if (Application.isPlaying)
                {
                    Destroy(col);
                }
                else
                {
                    DestroyImmediate(col);
                }
            }

            bar = go.transform;
        }

        bar.localPosition = localPos;
        bar.localRotation = Quaternion.identity;
        bar.localScale = localScale;

        Renderer r = bar.GetComponent<Renderer>();

        if (r != null)
        {
            if (_runtimeMaterial == null)
            {
                Shader shader = Shader.Find("Standard");

                if (shader == null)
                {
                    shader = Shader.Find("Universal Render Pipeline/Lit");
                }

                if (shader != null)
                {
                    _runtimeMaterial = new Material(shader);
                    _runtimeMaterial.color = frameColor;
                }
            }

            if (_runtimeMaterial != null)
            {
                r.sharedMaterial = _runtimeMaterial;
            }
        }
    }

    // Adds or updates the trigger box that matches the visible placement frame.
    // This provides a simple area volume without needing separate manually placed colliders.
    private void SetupTriggerCollider()
    {
        if (!addTriggerCollider)
        {
            return;
        }

        BoxCollider box = GetComponent<BoxCollider>();

        if (box == null)
        {
            box = gameObject.AddComponent<BoxCollider>();
        }

        box.isTrigger = true;
        box.center = new Vector3(0f, triggerHeight * 0.5f, 0f);
        box.size = new Vector3(frameSize, triggerHeight, frameSize);
    }

    // Computes horizontal distance while ignoring height.
    // This is used because placement reachability and spacing are mainly checked on the table plane.
    private float XZDistance(Vector3 a, Vector3 b)
    {
        Vector2 aa = new Vector2(a.x, a.z);
        Vector2 bb = new Vector2(b.x, b.z);
        return Vector2.Distance(aa, bb);
    }

    // Editor-only visual debugging.
    // The cyan box shows the randomisation area and current trigger frame, while the red/yellow spheres show
    // the minimum and maximum valid distances from the robot base.
    private void OnDrawGizmos()
    {
        if (!drawGizmos)
        {
            return;
        }

        Vector3 areaCenter = GetAreaCenterWorld();

        Gizmos.color = new Color(0f, 1f, 1f, 0.25f);
        Gizmos.DrawWireCube(
            areaCenter,
            new Vector3(areaHalfSizeX * 2f, 0.02f, areaHalfSizeZ * 2f)
        );

        Gizmos.color = new Color(0f, 1f, 1f, 0.7f);
        Gizmos.DrawWireCube(
            transform.position,
            new Vector3(frameSize, triggerHeight, frameSize)
        );

        if (robotBase != null)
        {
            Gizmos.color = new Color(1f, 0f, 0f, 0.25f);
            Gizmos.DrawWireSphere(
                new Vector3(robotBase.position.x, fixedY, robotBase.position.z),
                minDistanceFromRobot
            );

            Gizmos.color = new Color(1f, 1f, 0f, 0.20f);
            Gizmos.DrawWireSphere(
                new Vector3(robotBase.position.x, fixedY, robotBase.position.z),
                maxDistanceFromRobot
            );
        }
    }
}