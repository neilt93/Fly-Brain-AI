using UnityEngine;
using UnityEngine.InputSystem;

/// <summary>
/// Drives the fly animation from experiment data.
/// Supports both connectome demo mode (single fly + brain viz)
/// and comparison mode (plastic vs fixed fly).
/// </summary>
public class FlyAnimator : MonoBehaviour
{
    [Header("References")]
    public FlyDataLoader dataLoader;
    public MeshFly plasticFly;
    public MeshFly fixedFly;
    public ConnectomeViz connectomeViz;

    [Header("Mode")]
    public bool connectomeDemo = false;

    [Header("Playback")]
    public float playbackSpeed = 1f;
    public bool playing = true;
    public bool loop = true;

    // HUD state
    string frameLabel = "";
    string scoreLabel = "";
    string phaseLabel = "";
    string connectomeLabel = "";

    int currentFrame = 0;
    float frameTimer = 0f;
    int totalFrames = 0;

    // Cached joint names (same for all frames)
    string[] plasticJointNames;
    string[] fixedJointNames;

    // Camera focus tracking
    Transform cameraFocus;

    // Connectome state
    bool connectomeReady = false;

    void Start()
    {
        if (dataLoader == null)
            dataLoader = FindFirstObjectByType<FlyDataLoader>();

        if (!dataLoader.dataLoaded)
        {
            Debug.LogError("No data loaded!");
            return;
        }

        totalFrames = dataLoader.plasticData.n_frames;

        // Cache joint names
        plasticJointNames = dataLoader.GetJointNames(dataLoader.plasticData);
        fixedJointNames = dataLoader.fixedData != null ? dataLoader.GetJointNames(dataLoader.fixedData) : null;

        if (plasticFly != null)
            plasticFly.transform.position = new Vector3(0, 0, 0);
        if (fixedFly != null && !connectomeDemo)
            fixedFly.transform.position = new Vector3(3f, 0, 0);

        // Find camera focus point for dynamic tracking
        var focusObj = GameObject.Find("CameraFocus");
        if (focusObj != null) cameraFocus = focusObj.transform;

        // Initialize connectome viz if data is available
        if (connectomeDemo && connectomeViz != null && dataLoader.connectomeLoaded)
        {
            connectomeViz.Build(dataLoader.connectomeData);
            connectomeReady = true;
            Debug.Log("Connectome visualization initialized");
        }
    }

    void Update()
    {
        if (dataLoader == null || !dataLoader.dataLoaded || totalFrames == 0) return;

        HandleInput();

        if (playing)
        {
            frameTimer += Time.deltaTime * playbackSpeed;
            float frameDuration = dataLoader.plasticData.dt;
            if (frameDuration <= 0) frameDuration = 0.01f;

            while (frameTimer >= frameDuration)
            {
                frameTimer -= frameDuration;
                currentFrame++;

                if (currentFrame >= totalFrames)
                {
                    if (loop)
                        currentFrame = 0;
                    else
                    {
                        currentFrame = totalFrames - 1;
                        playing = false;
                        break;
                    }
                }
            }
        }

        UpdateFlies();
        UpdateConnectome();
        UpdateHUD();
    }

    void UpdateFlies()
    {
        if (plasticFly != null && dataLoader.plasticData != null)
        {
            Vector3 pos = dataLoader.GetPosition(dataLoader.plasticData, currentFrame);
            plasticFly.transform.position = new Vector3(pos.x, pos.y, pos.z);

            int[] contacts = dataLoader.GetContacts(dataLoader.plasticData, currentFrame);
            plasticFly.SetContacts(contacts);

            // Drive joint angles if available
            float[] angles = dataLoader.GetJointAngles(dataLoader.plasticData, currentFrame);
            if (angles != null && plasticJointNames != null)
                plasticFly.SetJointAngles(angles, plasticJointNames);

            float tripod = dataLoader.GetTripodScore(dataLoader.plasticData, currentFrame);
            plasticFly.SetTripodFeedback(tripod);

            float drift = dataLoader.GetWeightDrift(dataLoader.plasticData, currentFrame);
            plasticFly.SetNeuralActivity(Mathf.Clamp01(drift * 20f));
        }

        if (!connectomeDemo && fixedFly != null && dataLoader.fixedData != null)
        {
            Vector3 pos = dataLoader.GetPosition(dataLoader.fixedData, currentFrame);
            fixedFly.transform.position = new Vector3(pos.x + 3f, pos.y, pos.z);

            int[] contacts = dataLoader.GetContacts(dataLoader.fixedData, currentFrame);
            fixedFly.SetContacts(contacts);

            float[] angles = dataLoader.GetJointAngles(dataLoader.fixedData, currentFrame);
            if (angles != null && fixedJointNames != null)
                fixedFly.SetJointAngles(angles, fixedJointNames);

            float tripod = dataLoader.GetTripodScore(dataLoader.fixedData, currentFrame);
            fixedFly.SetTripodFeedback(tripod);

            fixedFly.SetNeuralActivity(0f);
        }

        // Update camera focus
        if (cameraFocus != null)
        {
            if (connectomeDemo && plasticFly != null)
            {
                // Focus on single fly, slightly above for brain viz
                Vector3 flyPos = plasticFly.transform.position;
                cameraFocus.position = new Vector3(flyPos.x, Mathf.Max(flyPos.y + 1.0f, 1.0f), flyPos.z);
            }
            else if (plasticFly != null && fixedFly != null)
            {
                Vector3 mid = (plasticFly.transform.position + fixedFly.transform.position) * 0.5f;
                mid.y = Mathf.Max(mid.y, 0.3f);
                cameraFocus.position = mid;
            }
        }
    }

    void UpdateConnectome()
    {
        if (!connectomeReady || connectomeViz == null) return;

        float[] rates = dataLoader.GetFiringRates(currentFrame);
        if (rates != null)
            connectomeViz.SetFrameData(rates);
    }

    void UpdateHUD()
    {
        frameLabel = $"Frame {currentFrame}/{totalFrames}";

        if (connectomeDemo)
        {
            float pScore = dataLoader.GetTripodScore(dataLoader.plasticData, currentFrame);
            scoreLabel = $"Tripod Score: {pScore:F2}";

            if (connectomeReady)
            {
                int activeCount = connectomeViz.GetActiveCount();
                connectomeLabel = $"Active: {activeCount}/{dataLoader.connectomeData.n_neurons} viz neurons";
            }
        }
        else
        {
            float pScore = dataLoader.GetTripodScore(dataLoader.plasticData, currentFrame);
            float fScore = dataLoader.fixedData != null ? dataLoader.GetTripodScore(dataLoader.fixedData, currentFrame) : 0f;
            scoreLabel = $"Tripod  Plastic: {pScore:F2}  Fixed: {fScore:F2}";
        }

        if (dataLoader.plasticData != null)
        {
            int pertIdx = dataLoader.plasticData.perturbation_idx ?? 0;
            phaseLabel = currentFrame < pertIdx ? "FLAT TERRAIN" : "BLOCKS TERRAIN";
        }
    }

    void OnGUI()
    {
        if (connectomeDemo)
            DrawConnectomeHUD();
        else
            DrawComparisonHUD();
    }

    void DrawConnectomeHUD()
    {
        float w = Screen.width;
        float h = Screen.height;

        var titleStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = 24,
            fontStyle = FontStyle.Bold,
            alignment = TextAnchor.UpperCenter
        };
        titleStyle.normal.textColor = new Color(0.6f, 0.85f, 1f);

        var subtitleStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = 14,
            fontStyle = FontStyle.Italic,
            alignment = TextAnchor.UpperCenter
        };
        subtitleStyle.normal.textColor = new Color(0.4f, 0.5f, 0.6f);

        var infoStyle = new GUIStyle(GUI.skin.label) { fontSize = 13 };
        infoStyle.normal.textColor = new Color(0.5f, 0.5f, 0.6f);

        var stimStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = 15,
            fontStyle = FontStyle.Bold,
            alignment = TextAnchor.UpperRight
        };
        stimStyle.normal.textColor = new Color(1f, 0.6f, 0.2f);

        var controlStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = 11,
            alignment = TextAnchor.LowerCenter
        };
        controlStyle.normal.textColor = new Color(0.3f, 0.3f, 0.4f);

        // Title
        GUI.Label(new Rect(0, 10, w, 35), "CONNECTOME FLY BRAIN", titleStyle);
        GUI.Label(new Rect(0, 40, w, 25),
            "139,000 neurons | FlyWire connectome | LIF dynamics", subtitleStyle);

        // Stimulation info
        GUI.Label(new Rect(w - 310, 10, 300, 25), "Sensory Stimulation: Active", stimStyle);

        // Bottom info
        GUI.Label(new Rect(10, h - 50, 300, 25), frameLabel, infoStyle);
        GUI.Label(new Rect(10, h - 30, 400, 25), connectomeLabel, infoStyle);
        GUI.Label(new Rect(w - 310, h - 30, 300, 25), scoreLabel, infoStyle);

        // Phase
        var phaseStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = 15,
            fontStyle = FontStyle.Bold,
            alignment = TextAnchor.UpperRight
        };
        phaseStyle.normal.textColor = new Color(0.4f, 0.8f, 0.4f);
        GUI.Label(new Rect(w - 210, 35, 200, 25), phaseLabel, phaseStyle);

        // Controls
        GUI.Label(new Rect(0, h - 18, w, 20),
            "[Space] Play/Pause   [R] Reset   [Arrows] Speed/Step   [RMB] Orbit   [Scroll] Zoom   [MMB] Pan   [F] Recenter",
            controlStyle);
    }

    void DrawComparisonHUD()
    {
        var titleStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = 22,
            fontStyle = FontStyle.Bold,
            alignment = TextAnchor.UpperCenter
        };
        titleStyle.normal.textColor = new Color(0.6f, 0.8f, 1f);

        var subtitleStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = 14,
            fontStyle = FontStyle.Italic,
            alignment = TextAnchor.UpperCenter
        };
        subtitleStyle.normal.textColor = new Color(0.5f, 0.5f, 0.6f);

        var infoStyle = new GUIStyle(GUI.skin.label) { fontSize = 14 };
        infoStyle.normal.textColor = new Color(0.5f, 0.5f, 0.6f);

        var phaseStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = 16,
            fontStyle = FontStyle.Bold,
            alignment = TextAnchor.UpperRight
        };
        phaseStyle.normal.textColor = new Color(0.4f, 0.8f, 0.4f);

        var controlStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = 11,
            alignment = TextAnchor.LowerCenter
        };
        controlStyle.normal.textColor = new Color(0.3f, 0.3f, 0.4f);

        float w = Screen.width;
        float h = Screen.height;

        GUI.Label(new Rect(0, 10, w, 35), "CONNECTOME FLY BRAIN", titleStyle);
        GUI.Label(new Rect(0, 40, w, 25), "structure encodes meaning", subtitleStyle);
        GUI.Label(new Rect(10, h - 30, 300, 25), frameLabel, infoStyle);
        GUI.Label(new Rect(w - 310, h - 30, 300, 25), scoreLabel, infoStyle);
        GUI.Label(new Rect(w - 210, 10, 200, 30), phaseLabel, phaseStyle);
        GUI.Label(new Rect(0, h - 25, w, 20),
            "[Space] Play/Pause   [R] Reset   [Arrows] Speed/Step   [RMB] Orbit   [Scroll] Zoom   [MMB] Pan   [F] Recenter",
            controlStyle);
    }

    void HandleInput()
    {
        var kb = Keyboard.current;
        if (kb == null) return;

        if (kb.spaceKey.wasPressedThisFrame)
            playing = !playing;

        if (kb.rKey.wasPressedThisFrame)
            currentFrame = 0;

        if (kb.rightArrowKey.wasPressedThisFrame && !playing)
            currentFrame = Mathf.Min(currentFrame + 1, totalFrames - 1);

        if (kb.leftArrowKey.wasPressedThisFrame && !playing)
            currentFrame = Mathf.Max(currentFrame - 1, 0);

        if (kb.upArrowKey.wasPressedThisFrame)
            playbackSpeed = Mathf.Min(playbackSpeed * 2f, 16f);

        if (kb.downArrowKey.wasPressedThisFrame)
            playbackSpeed = Mathf.Max(playbackSpeed * 0.5f, 0.125f);
    }
}
