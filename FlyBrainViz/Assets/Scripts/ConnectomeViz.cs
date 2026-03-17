using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Visualizes real FlyWire connectome neural activity above the fly.
/// Reads connectome_activity.json: 200-300 neuron nodes with real synaptic connections,
/// firing rates per frame from a Brian2 LIF simulation of 139k neurons.
/// Replaces NeuralNetworkViz with actual connectome data.
/// </summary>
public class ConnectomeViz : MonoBehaviour
{
    [Header("Layout")]
    public Vector3 offset = new Vector3(0, 2.0f, 0);
    public float brainScale = 1.5f;

    [Header("Node Sizing")]
    public float baseNodeSize = 0.025f;
    public float stimulatedNodeSize = 0.05f;
    public float pulseAmplitude = 0.4f;

    [Header("Connection")]
    public float connectionWidth = 0.004f;
    public float connectionActiveWidth = 0.012f;

    [Header("Colors by Type")]
    public Color stimulatedColor = new Color(1f, 0.6f, 0.1f);       // orange — sensory input
    public Color descendingColor = new Color(0.9f, 0.2f, 0.15f);    // red — motor-adjacent
    public Color hubColor = new Color(0.3f, 0.5f, 1f);              // blue — high-connectivity
    public Color interneuronColor = new Color(0.2f, 0.8f, 0.5f);    // green — interneurons
    public Color peripheralColor = new Color(0.4f, 0.3f, 0.6f);     // purple — low-activity

    public Color connectionInactive = new Color(0.15f, 0.2f, 0.35f, 0.15f);
    public Color connectionActive = new Color(0.6f, 0.7f, 1f, 0.6f);

    [Header("Animation")]
    public float rotationSpeed = 8f;  // degrees per second
    public float activitySmoothing = 5f;

    // Internal state
    GameObject[] nodes;
    Material[] nodeMats;
    float[] currentActivity;   // smoothed per-node activity
    float[] nodeBaseSize;
    Color[] nodeBaseColor;

    LineRenderer[] connectionLines;
    int[] connFrom;
    int[] connTo;
    float[] connWeight;

    int nNeurons;
    int nConnections;
    bool initialized = false;

    /// <summary>
    /// Build the visualization from loaded connectome data.
    /// Call this once after FlyDataLoader has parsed the JSON.
    /// </summary>
    public void Build(FlyDataLoader.ConnectomeData data)
    {
        if (data == null)
        {
            Debug.LogError("ConnectomeViz.Build: null data");
            return;
        }

        transform.localPosition = offset;

        nNeurons = data.n_neurons;
        nodes = new GameObject[nNeurons];
        nodeMats = new Material[nNeurons];
        currentActivity = new float[nNeurons];
        nodeBaseSize = new float[nNeurons];
        nodeBaseColor = new Color[nNeurons];

        // Guard: ensure positions_3d has enough entries
        if (data.positions_3d == null || data.positions_3d.Count < nNeurons)
        {
            Debug.LogError($"ConnectomeViz.Build: positions_3d has {data.positions_3d?.Count ?? 0} entries, need {nNeurons}");
            return;
        }

        // Create neuron spheres
        for (int i = 0; i < nNeurons; i++)
        {
            var pos3 = data.positions_3d[i];
            Vector3 pos;
            if (pos3 == null || pos3.Count < 3)
            {
                Debug.LogWarning($"ConnectomeViz: position {i} has {pos3?.Count ?? 0} elements, using origin");
                pos = Vector3.zero;
            }
            else
            {
                pos = new Vector3(pos3[0], pos3[1], pos3[2]) * brainScale;
            }

            var node = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            node.name = i < data.neuron_names.Count ? data.neuron_names[i] : $"N_{i}";
            node.transform.SetParent(transform);
            node.transform.localPosition = pos;

            // Remove collider for performance
            Object.Destroy(node.GetComponent<Collider>());

            // Color by type
            string ntype = i < data.neuron_types.Count ? data.neuron_types[i] : "interneuron";
            Color baseColor = GetTypeColor(ntype);
            nodeBaseColor[i] = baseColor;

            float size = ntype == "stimulated" ? stimulatedNodeSize : baseNodeSize;
            nodeBaseSize[i] = size;
            node.transform.localScale = Vector3.one * size;

            // Material with emission
            var mat = new Material(Shader.Find("Standard"));
            mat.EnableKeyword("_EMISSION");
            mat.color = baseColor * 0.5f;
            mat.SetColor("_EmissionColor", baseColor * 0.2f);
            node.GetComponent<Renderer>().material = mat;

            nodes[i] = node;
            nodeMats[i] = mat;
        }

        // Create connection lines
        if (data.connections != null && data.connections.Count > 0)
        {
            nConnections = data.connections.Count;
            connectionLines = new LineRenderer[nConnections];
            connFrom = new int[nConnections];
            connTo = new int[nConnections];
            connWeight = new float[nConnections];

            // Find max weight for normalization
            float maxW = 0f;
            for (int c = 0; c < nConnections; c++)
            {
                if (data.connections[c] == null || data.connections[c].Count < 3)
                    continue;
                float w = Mathf.Abs(data.connections[c][2]);
                if (w > maxW) maxW = w;
            }
            if (maxW == 0) maxW = 1f;

            for (int c = 0; c < nConnections; c++)
            {
                if (data.connections[c] == null || data.connections[c].Count < 3)
                    continue;
                int from = (int)data.connections[c][0];
                int to = (int)data.connections[c][1];
                if (from < 0 || from >= nNeurons || to < 0 || to >= nNeurons)
                    continue;
                float w = data.connections[c][2] / maxW; // normalized

                connFrom[c] = from;
                connTo[c] = to;
                connWeight[c] = w;

                var connObj = new GameObject($"Conn_{from}_{to}");
                connObj.transform.SetParent(transform);

                var lr = connObj.AddComponent<LineRenderer>();
                lr.positionCount = 2;
                lr.startWidth = connectionWidth;
                lr.endWidth = connectionWidth;
                lr.material = new Material(Shader.Find("Sprites/Default"));
                lr.startColor = connectionInactive;
                lr.endColor = connectionInactive;
                lr.useWorldSpace = false;

                lr.SetPosition(0, nodes[from].transform.localPosition);
                lr.SetPosition(1, nodes[to].transform.localPosition);

                connectionLines[c] = lr;
            }
        }
        else
        {
            nConnections = 0;
            connectionLines = new LineRenderer[0];
            connFrom = new int[0];
            connTo = new int[0];
            connWeight = new float[0];
        }

        initialized = true;
        Debug.Log($"ConnectomeViz built: {nNeurons} neurons, {nConnections} connections");
    }

    Color GetTypeColor(string ntype)
    {
        switch (ntype)
        {
            case "stimulated": return stimulatedColor;
            case "descending": return descendingColor;
            case "hub": return hubColor;
            case "interneuron": return interneuronColor;
            case "peripheral": return peripheralColor;
            default: return interneuronColor;
        }
    }

    /// <summary>
    /// Set firing rates for current frame. Called by FlyAnimator each update.
    /// rates array length must match n_neurons.
    /// </summary>
    public void SetFrameData(float[] rates)
    {
        if (!initialized || rates == null) return;

        int n = Mathf.Min(rates.Length, nNeurons);
        float dt = Time.deltaTime * activitySmoothing;

        for (int i = 0; i < n; i++)
        {
            // Smooth transition
            currentActivity[i] = Mathf.Lerp(currentActivity[i], rates[i], dt);
        }
    }

    void Update()
    {
        if (!initialized) return;

        // Update node visuals
        for (int i = 0; i < nNeurons; i++)
        {
            float a = currentActivity[i];
            Color base_c = nodeBaseColor[i];

            // Color: blend from dim to full based on activity
            Color c = Color.Lerp(base_c * 0.3f, base_c, a);
            nodeMats[i].color = c;
            nodeMats[i].SetColor("_EmissionColor", base_c * a * 2f);

            // Size pulse
            float size = nodeBaseSize[i] * (1f + a * pulseAmplitude);
            nodes[i].transform.localScale = Vector3.one * size;
        }

        // Update connections
        for (int c = 0; c < nConnections; c++)
        {
            if (connFrom[c] < 0 || connFrom[c] >= nNeurons || connTo[c] < 0 || connTo[c] >= nNeurons)
                continue;
            float avgAct = (currentActivity[connFrom[c]] + currentActivity[connTo[c]]) * 0.5f;
            float w = Mathf.Abs(connWeight[c]);
            float intensity = avgAct * w;

            Color col = Color.Lerp(connectionInactive, connectionActive, intensity);
            connectionLines[c].startColor = col;
            connectionLines[c].endColor = col;

            float width = Mathf.Lerp(connectionWidth, connectionActiveWidth, intensity);
            connectionLines[c].startWidth = width;
            connectionLines[c].endWidth = width;
        }

        // Gentle rotation
        transform.Rotate(Vector3.up, rotationSpeed * Time.deltaTime, Space.Self);
    }

    /// <summary>
    /// Count currently active neurons (activity > threshold).
    /// </summary>
    public int GetActiveCount(float threshold = 0.1f)
    {
        if (!initialized) return 0;
        int count = 0;
        for (int i = 0; i < nNeurons; i++)
            if (currentActivity[i] > threshold)
                count++;
        return count;
    }
}
