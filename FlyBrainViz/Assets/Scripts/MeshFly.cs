using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Anatomical Drosophila controller — replaces ProceduralFly.
/// Builds the real NeuroMechFly model from MJCF + STL meshes,
/// then drives joint rotations from experiment data each frame.
/// </summary>
public class MeshFly : MonoBehaviour
{
    [Header("Scale")]
    public float globalScale = 0.5f; // mm → Unity units

    [Header("Colors")]
    public Color legStanceColor = new Color(0.1f, 0.9f, 0.3f);
    public Color legSwingColor = new Color(0.9f, 0.15f, 0.1f);
    public Color neuralGlow = new Color(1f, 0.3f, 0.1f);

    [Header("Neural Viz")]
    public float glowIntensity = 2f;

    // Built model data
    List<JointInfo> animatedBodies;
    Dictionary<string, List<Renderer>> legRenderers;
    Renderer thoraxRenderer;
    List<Renderer> abdomenRenderers;

    // Leg color tracking
    Dictionary<string, List<Material>> legMaterials;
    Material thoraxMat;
    List<Material> abdomenMats;

    // Reusable angle dict for joint driving
    Dictionary<string, float> angleDict = new Dictionary<string, float>();

    // Leg prefix order matching contacts array: LF=0, LM=1, LH=2, RF=3, RM=4, RH=5
    static readonly string[] LEG_PREFIXES = { "LF", "LM", "LH", "RF", "RM", "RH" };

    void Start()
    {
        Build();
    }

    void Build()
    {
        var result = MjcfFlyBuilder.BuildFly(transform, globalScale);
        if (result == null || result.root == null)
        {
            Debug.LogError("MeshFly: Failed to build fly model!");
            return;
        }

        animatedBodies = result.animatedBodies;
        legRenderers = result.legRenderers;
        thoraxRenderer = result.thoraxRenderer;
        abdomenRenderers = result.abdomenRenderers;

        // Cache materials for runtime color changes
        legMaterials = new Dictionary<string, List<Material>>();
        foreach (var prefix in LEG_PREFIXES)
        {
            var mats = new List<Material>();
            if (legRenderers.ContainsKey(prefix))
            {
                foreach (var r in legRenderers[prefix])
                    mats.Add(r.material);
            }
            legMaterials[prefix] = mats;
        }

        if (thoraxRenderer != null)
            thoraxMat = thoraxRenderer.material;

        abdomenMats = new List<Material>();
        foreach (var r in abdomenRenderers)
            abdomenMats.Add(r.material);
    }

    /// <summary>
    /// Drive all joints from an array of 42 angles (radians) with matching joint names.
    /// </summary>
    public void SetJointAngles(float[] angles, string[] jointNames)
    {
        if (animatedBodies == null || angles == null || jointNames == null) return;

        // Build angle dictionary
        angleDict.Clear();
        int count = Mathf.Min(angles.Length, jointNames.Length);
        for (int i = 0; i < count; i++)
            angleDict[jointNames[i]] = angles[i];

        // Apply to all animated bodies
        foreach (var ji in animatedBodies)
            ji.ApplyAngles(angleDict);
    }

    /// <summary>
    /// Update leg colors based on contact state.
    /// contacts[i]: 1 = stance, 0 = swing. Order: LF, LM, LH, RF, RM, RH.
    /// </summary>
    public void SetContacts(int[] contacts)
    {
        if (legMaterials == null || contacts == null) return;

        for (int i = 0; i < 6 && i < contacts.Length; i++)
        {
            bool stance = contacts[i] == 1;
            Color target = stance ? legStanceColor : legSwingColor;
            float emission = stance ? 0.4f : 0.15f;

            if (legMaterials.TryGetValue(LEG_PREFIXES[i], out var mats))
            {
                foreach (var mat in mats)
                {
                    mat.color = Color.Lerp(mat.color, target, Time.deltaTime * 10f);
                    mat.SetColor("_EmissionColor", target * emission);
                }
            }
        }
    }

    /// <summary>
    /// Pulse thorax/abdomen glow based on neural/plasticity activity.
    /// </summary>
    public void SetNeuralActivity(float activity)
    {
        float pulse = Mathf.Sin(Time.time * 3f) * 0.5f + 0.5f;
        float intensity = activity * glowIntensity * (0.5f + pulse * 0.5f);
        Color emission = neuralGlow * intensity;

        if (thoraxMat != null)
            thoraxMat.SetColor("_EmissionColor", emission);

        foreach (var mat in abdomenMats)
            mat.SetColor("_EmissionColor", emission * 0.5f);
    }

    /// <summary>
    /// Tint thorax based on tripod coordination quality.
    /// </summary>
    public void SetTripodFeedback(float tripodScore)
    {
        if (thoraxMat == null) return;

        Color tint = Color.Lerp(
            new Color(0.4f, 0.05f, 0.05f),
            new Color(0.05f, 0.12f, 0.35f),
            tripodScore
        );
        Color baseColor = new Color(0.59f, 0.39f, 0.12f);
        thoraxMat.color = Color.Lerp(baseColor, tint, 0.25f);
    }
}
