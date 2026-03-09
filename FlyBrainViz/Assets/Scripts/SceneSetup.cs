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
        SetupGround();

        if (connectomeDemo)
            SetupConnectomeDemo();
        else
            SetupComparisonMode();

        SetupCamera();

        // Dark background
        Camera.main.backgroundColor = new Color(0.02f, 0.02f, 0.06f);
        Camera.main.clearFlags = CameraClearFlags.SolidColor;
    }

    void SetupLighting()
    {
        var lightObj = new GameObject("MainLight");
        var light = lightObj.AddComponent<Light>();
        light.type = LightType.Directional;
        light.color = new Color(0.9f, 0.85f, 0.8f);
        light.intensity = 1.2f;
        light.shadows = LightShadows.Soft;
        lightObj.transform.rotation = Quaternion.Euler(45, -30, 0);

        var fillObj = new GameObject("FillLight");
        var fill = fillObj.AddComponent<Light>();
        fill.type = LightType.Directional;
        fill.color = new Color(0.3f, 0.4f, 0.7f);
        fill.intensity = 0.4f;
        fillObj.transform.rotation = Quaternion.Euler(30, 150, 0);

        var rimObj = new GameObject("RimLight");
        var rim = rimObj.AddComponent<Light>();
        rim.type = LightType.Directional;
        rim.color = new Color(1f, 0.5f, 0.2f);
        rim.intensity = 0.6f;
        rimObj.transform.rotation = Quaternion.Euler(10, -120, 0);

        RenderSettings.ambientLight = new Color(0.05f, 0.05f, 0.1f);
    }

    void SetupGround()
    {
        var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
        ground.name = "Ground";
        ground.transform.position = new Vector3(0f, -0.8f, 2f);
        ground.transform.localScale = new Vector3(3f, 1f, 10f);

        var mat = new Material(Shader.Find("Standard"));
        mat.color = new Color(0.08f, 0.08f, 0.12f);
        mat.SetFloat("_Metallic", 0.3f);
        mat.SetFloat("_Glossiness", 0.7f);
        ground.GetComponent<Renderer>().material = mat;

        // Grid lines
        for (int i = 0; i < 40; i++)
        {
            var line = new GameObject($"GridLine_{i}");
            line.transform.SetParent(ground.transform);
            var lr = line.AddComponent<LineRenderer>();
            lr.positionCount = 2;
            lr.startWidth = 0.005f;
            lr.endWidth = 0.005f;
            lr.material = new Material(Shader.Find("Sprites/Default"));
            lr.startColor = new Color(0.15f, 0.15f, 0.25f, 0.3f);
            lr.endColor = new Color(0.15f, 0.15f, 0.25f, 0.3f);

            float z = i * 0.5f - 5f;
            lr.SetPosition(0, new Vector3(-3f, -0.78f, z));
            lr.SetPosition(1, new Vector3(6f, -0.78f, z));
        }

        // Terrain transition marker
        var marker = new GameObject("BlocksMarker");
        var mlr = marker.AddComponent<LineRenderer>();
        mlr.positionCount = 2;
        mlr.startWidth = 0.02f;
        mlr.endWidth = 0.02f;
        mlr.material = new Material(Shader.Find("Sprites/Default"));
        mlr.startColor = new Color(1f, 0.3f, 0.1f, 0.6f);
        mlr.endColor = new Color(1f, 0.3f, 0.1f, 0.6f);
        float blocksZ = 3f;
        mlr.SetPosition(0, new Vector3(-3f, -0.75f, blocksZ));
        mlr.SetPosition(1, new Vector3(6f, -0.75f, blocksZ));
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
        anim.playbackSpeed = 0.1f;

        // Labels
        CreateWorldLabel("WHOLE-BRAIN EMULATION", new Vector3(0, 3.8f, -0.5f),
            new Color(0.6f, 0.8f, 1f));
        CreateWorldLabel("139,000 neurons | FlyWire Connectome", new Vector3(0, 3.5f, -0.5f),
            new Color(0.4f, 0.5f, 0.6f), fontSize: 30);
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
        anim.playbackSpeed = 0.1f;

        // World-space labels
        CreateWorldLabel("PLASTIC (adapting)", new Vector3(0, 2f, -0.5f),
            new Color(1f, 0.4f, 0.1f));
        CreateWorldLabel("FIXED (frozen)", new Vector3(3f, 2f, -0.5f),
            new Color(0.5f, 0.5f, 0.7f));
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

    void SetupCamera()
    {
        var cam = Camera.main;
        if (cam == null)
        {
            var camObj = new GameObject("MainCamera");
            cam = camObj.AddComponent<Camera>();
        }

        // Focus point
        var focusObj = new GameObject("CameraFocus");
        if (connectomeDemo)
            focusObj.transform.position = new Vector3(0, 1.5f, 0);
        else
            focusObj.transform.position = new Vector3(1.5f, 0.5f, 0);

        // Interactive orbit camera
        var orbit = cam.gameObject.AddComponent<OrbitCamera>();
        orbit.target = focusObj.transform;
        orbit.distance = connectomeDemo ? 5f : 4f;
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
