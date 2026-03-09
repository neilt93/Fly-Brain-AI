using UnityEngine;

/// <summary>
/// Auto-creates the demo scene on play.
/// Attach to any GameObject in the default scene, or use RuntimeInitializeOnLoadMethod.
/// </summary>
public class AutoBootstrap
{
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void OnSceneLoad()
    {
        // Only bootstrap if SceneSetup doesn't already exist
        if (Object.FindFirstObjectByType<SceneSetup>() != null) return;

        Debug.Log("FlyBrainViz: Auto-bootstrapping demo scene...");
        var setupObj = new GameObject("SceneSetup");
        setupObj.AddComponent<SceneSetup>();
    }
}
