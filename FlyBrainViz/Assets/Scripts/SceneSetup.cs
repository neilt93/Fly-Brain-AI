using UnityEngine;

/// <summary>
/// Bootstraps the entire demo scene procedurally.
/// Supports two modes:
/// - Connectome demo (default): single fly with real connectome brain visualization
/// - Comparison mode: two flies (plastic vs fixed) with simple neural viz
/// No UI module dependency — uses TextMesh and OnGUI.
/// </summary>
public class SceneSetup : MonoBehaviour
{
    [Header("Mode")]
    public bool connectomeDemo = true;  // true = single fly + connectome brain viz

    void Start()
    {
        SetupLighting();
        SetupEnvironment();

        if (connectomeDemo)
            SetupConnectomeDemo();
        else
            SetupComparisonMode();

        Camera cam = SetupCamera();

        // Sky background
        if (cam != null)
        {
            cam.backgroundColor = new Color(0.53f, 0.75f, 0.92f);
            cam.clearFlags = CameraClearFlags.SolidColor;
        }
    }

    void SetupLighting()
    {
        // Sun — warm directional from upper-left
        var sunObj = new GameObject("Sun");
        var sun = sunObj.AddComponent<Light>();
        sun.type = LightType.Directional;
        sun.color = new Color(1f, 0.95f, 0.84f);
        sun.intensity = 1.4f;
        sun.shadows = LightShadows.Soft;
        sun.shadowStrength = 0.5f;
        sunObj.transform.rotation = Quaternion.Euler(50, -40, 0);

        // Fill — cool sky light from opposite side
        var fillObj = new GameObject("FillLight");
        var fill = fillObj.AddComponent<Light>();
        fill.type = LightType.Directional;
        fill.color = new Color(0.55f, 0.65f, 0.85f);
        fill.intensity = 0.5f;
        fill.shadows = LightShadows.None;
        fillObj.transform.rotation = Quaternion.Euler(25, 140, 0);

        // Warm ambient — outdoor feel
        RenderSettings.ambientLight = new Color(0.22f, 0.20f, 0.16f);

        // Distance fog
        RenderSettings.fog = true;
        RenderSettings.fogMode = FogMode.Linear;
        RenderSettings.fogColor = new Color(0.6f, 0.72f, 0.82f);
        RenderSettings.fogStartDistance = 8f;
        RenderSettings.fogEndDistance = 35f;
    }

    void SetupEnvironment()
    {
        var envParent = new GameObject("Environment");
        var rng = new System.Random(42);

        // --- Main ground plane at Y=0 ---
        var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
        ground.name = "Ground";
        ground.transform.SetParent(envParent.transform);
        ground.transform.position = new Vector3(0f, 0f, 0f);
        ground.transform.localScale = new Vector3(6f, 1f, 6f); // 60x60 units

        var groundMat = new Material(Shader.Find("Standard"));
        groundMat.color = new Color(0.38f, 0.30f, 0.20f); // earthy brown
        groundMat.SetFloat("_Metallic", 0f);
        groundMat.SetFloat("_Glossiness", 0.08f);
        ground.GetComponent<Renderer>().material = groundMat;

        // --- Pebbles scattered on ground ---
        for (int i = 0; i < 30; i++)
        {
            var pebble = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            pebble.name = $"Pebble_{i}";
            pebble.transform.SetParent(envParent.transform);
            Object.Destroy(pebble.GetComponent<Collider>());

            float x = (float)(rng.NextDouble() * 8 - 4);
            float z = (float)(rng.NextDouble() * 8 - 4);
            float size = 0.02f + (float)(rng.NextDouble() * 0.07f);
            float ySquash = 0.3f + (float)(rng.NextDouble() * 0.4f);

            pebble.transform.position = new Vector3(x, size * ySquash * 0.4f, z);
            pebble.transform.localScale = new Vector3(size, size * ySquash, size);
            pebble.transform.rotation = Quaternion.Euler(
                (float)(rng.NextDouble() * 30 - 15),
                (float)(rng.NextDouble() * 360),
                (float)(rng.NextDouble() * 30 - 15));

            var mat = new Material(Shader.Find("Standard"));
            float shade = 0.22f + (float)(rng.NextDouble() * 0.18f);
            mat.color = new Color(shade, shade * 0.88f, shade * 0.72f);
            mat.SetFloat("_Metallic", 0f);
            mat.SetFloat("_Glossiness", 0.15f + (float)(rng.NextDouble() * 0.2f));
            pebble.GetComponent<Renderer>().material = mat;
        }

        // --- Grass blades (tall thin cubes around the perimeter) ---
        for (int i = 0; i < 18; i++)
        {
            float angle = i * Mathf.PI * 2f / 18f + (float)(rng.NextDouble() * 0.3f);
            float dist = 3.5f + (float)(rng.NextDouble() * 5f);
            float x = Mathf.Cos(angle) * dist;
            float z = Mathf.Sin(angle) * dist;
            float height = 2f + (float)(rng.NextDouble() * 5f);
            float width = 0.03f + (float)(rng.NextDouble() * 0.03f);
            float depth = 0.008f + (float)(rng.NextDouble() * 0.008f);

            var blade = GameObject.CreatePrimitive(PrimitiveType.Cube);
            blade.name = $"Grass_{i}";
            blade.transform.SetParent(envParent.transform);
            Object.Destroy(blade.GetComponent<Collider>());

            blade.transform.position = new Vector3(x, height * 0.5f, z);
            blade.transform.localScale = new Vector3(width, height, depth);
            // Slight lean outward + random twist
            Vector3 outDir = new Vector3(x, 0, z).normalized;
            float lean = 5f + (float)(rng.NextDouble() * 12f);
            blade.transform.rotation = Quaternion.LookRotation(outDir) *
                Quaternion.Euler(-lean, (float)(rng.NextDouble() * 20 - 10), 0);

            var mat = new Material(Shader.Find("Standard"));
            float g = 0.30f + (float)(rng.NextDouble() * 0.18f);
            mat.color = new Color(g * 0.45f, g, g * 0.25f);
            mat.SetFloat("_Metallic", 0f);
            mat.SetFloat("_Glossiness", 0.05f);
            blade.GetComponent<Renderer>().material = mat;
        }

        // --- A few small mounds for terrain variation ---
        for (int i = 0; i < 5; i++)
        {
            var mound = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            mound.name = $"Mound_{i}";
            mound.transform.SetParent(envParent.transform);
            Object.Destroy(mound.GetComponent<Collider>());

            float x = (float)(rng.NextDouble() * 6 - 3);
            float z = (float)(rng.NextDouble() * 6 - 3);
            float size = 0.15f + (float)(rng.NextDouble() * 0.25f);

            mound.transform.position = new Vector3(x, size * 0.15f, z);
            mound.transform.localScale = new Vector3(size, size * 0.3f, size);

            var mat = new Material(Shader.Find("Standard"));
            float shade = 0.32f + (float)(rng.NextDouble() * 0.1f);
            mat.color = new Color(shade, shade * 0.85f, shade * 0.65f);
            mat.SetFloat("_Metallic", 0f);
            mat.SetFloat("_Glossiness", 0.06f);
            mound.GetComponent<Renderer>().material = mat;
        }
    }

    void SetupConnectomeDemo()
    {
        // Data loader
        var loaderObj = new GameObject("DataLoader");
        var loader = loaderObj.AddComponent<FlyDataLoader>();

        // Single fly (centered)
        var flyObj = new GameObject("PlasticFly");
        flyObj.transform.position = new Vector3(0, 0, 0);
        var fly = flyObj.AddComponent<MeshFly>();
        fly.neuralGlow = new Color(1f, 0.3f, 0.1f);

        // Connectome visualization above fly
        var connectomeObj = new GameObject("ConnectomeViz");
        connectomeObj.transform.SetParent(flyObj.transform);
        var connectomeViz = connectomeObj.AddComponent<ConnectomeViz>();

        // Animator (single fly mode)
        var animObj = new GameObject("Animator");
        var anim = animObj.AddComponent<FlyAnimator>();
        anim.dataLoader = loader;
        anim.plasticFly = fly;
        anim.fixedFly = null;  // no fixed fly in connectome demo
        anim.connectomeViz = connectomeViz;
        anim.connectomeDemo = true;
        anim.playbackSpeed = 0.5f;

        // Labels
        CreateWorldLabel("WHOLE-BRAIN EMULATION", new Vector3(0, 2.8f, -0.5f),
            new Color(0.15f, 0.25f, 0.45f));
        CreateWorldLabel("139,000 neurons | FlyWire Connectome", new Vector3(0, 2.55f, -0.5f),
            new Color(0.3f, 0.35f, 0.4f), fontSize: 30);
    }

    void SetupComparisonMode()
    {
        // Data loader
        var loaderObj = new GameObject("DataLoader");
        var loader = loaderObj.AddComponent<FlyDataLoader>();

        // Plastic fly (anatomical mesh model)
        var plasticObj = new GameObject("PlasticFly");
        plasticObj.transform.position = new Vector3(0, 0, 0);
        var pFly = plasticObj.AddComponent<MeshFly>();
        pFly.neuralGlow = new Color(1f, 0.3f, 0.1f);

        // Neural network viz above plastic fly
        var neuralObj = new GameObject("NeuralViz");
        neuralObj.transform.SetParent(plasticObj.transform);
        neuralObj.AddComponent<NeuralNetworkViz>();

        // Fixed fly (anatomical mesh model)
        var fixedObj = new GameObject("FixedFly");
        fixedObj.transform.position = new Vector3(3f, 0, 0);
        var fFly = fixedObj.AddComponent<MeshFly>();
        fFly.neuralGlow = new Color(0.3f, 0.3f, 0.5f);

        // Animator
        var animObj = new GameObject("Animator");
        var anim = animObj.AddComponent<FlyAnimator>();
        anim.dataLoader = loader;
        anim.plasticFly = pFly;
        anim.fixedFly = fFly;
        anim.connectomeDemo = false;
        anim.playbackSpeed = 0.5f;

        // World-space labels
        CreateWorldLabel("PLASTIC (adapting)", new Vector3(0, 1.5f, -0.5f),
            new Color(0.7f, 0.25f, 0.05f));
        CreateWorldLabel("FIXED (frozen)", new Vector3(3f, 1.5f, -0.5f),
            new Color(0.3f, 0.3f, 0.5f));
    }

    void CreateWorldLabel(string text, Vector3 position, Color color, int fontSize = 40)
    {
        var obj = new GameObject($"Label_{text}");
        obj.transform.position = position;

        var tm = obj.AddComponent<TextMesh>();
        tm.text = text;
        tm.fontSize = fontSize;
        tm.characterSize = 0.05f;
        tm.anchor = TextAnchor.MiddleCenter;
        tm.alignment = TextAlignment.Center;
        tm.color = color;
        tm.fontStyle = FontStyle.Bold;

        obj.AddComponent<Billboard>();
    }

    Camera SetupCamera()
    {
        var cam = Camera.main;
        if (cam == null)
            cam = FindFirstObjectByType<Camera>();
        if (cam == null)
        {
            var camObj = new GameObject("MainCamera");
            cam = camObj.AddComponent<Camera>();
            camObj.tag = "MainCamera";
        }
        else if (!cam.CompareTag("MainCamera"))
            cam.tag = "MainCamera";

        // Focus point (adjusted for globalScale=0.5 position scaling)
        var focusObj = new GameObject("CameraFocus");
        if (connectomeDemo)
            focusObj.transform.position = new Vector3(0, 1.2f, 0);
        else
            focusObj.transform.position = new Vector3(1.5f, 0.4f, 0);

        // Interactive orbit camera
        var orbit = cam.gameObject.AddComponent<OrbitCamera>();
        orbit.target = focusObj.transform;
        orbit.distance = connectomeDemo ? 4f : 3.5f;

        return cam;
    }
}

/// <summary>
/// Simple billboard — always faces main camera.
/// </summary>
public class Billboard : MonoBehaviour
{
    void LateUpdate()
    {
        if (Camera.main != null)
            transform.forward = Camera.main.transform.forward;
    }
}
