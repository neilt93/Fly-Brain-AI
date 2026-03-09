using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Renders a floating neural network visualization above the fly.
/// Shows sparse recurrent connections with activity pulses.
/// The "structure encodes meaning" visual — connections light up
/// to show that topology matters, not just weights.
/// </summary>
public class NeuralNetworkViz : MonoBehaviour
{
    [Header("Network")]
    public int numNodes = 16;        // visual subset of 64-dim hidden
    public float networkRadius = 0.8f;
    public float nodeSize = 0.04f;
    public float connectionWidth = 0.01f;
    public float sparsity = 0.8f;    // fraction of connections pruned

    [Header("Positioning")]
    public Vector3 offset = new Vector3(0, 1.5f, 0);

    [Header("Colors")]
    public Color nodeInactive = new Color(0.2f, 0.2f, 0.3f);
    public Color nodeActive = new Color(1f, 0.4f, 0.1f);
    public Color connectionColor = new Color(0.3f, 0.5f, 1f, 0.3f);
    public Color connectionActive = new Color(1f, 0.6f, 0.2f, 0.8f);

    [Header("Animation")]
    public float pulseSpeed = 2f;
    public float activityDecay = 3f;

    // Internal
    GameObject[] nodes;
    Renderer[] nodeRenderers;
    Material[] nodeMats;
    LineRenderer[] connections;
    bool[,] connectionMask;
    float[] nodeActivities;
    float globalActivity = 0f;

    void Start()
    {
        transform.localPosition = offset;
        BuildNetwork();
    }

    void BuildNetwork()
    {
        // Create nodes in a ring layout
        nodes = new GameObject[numNodes];
        nodeRenderers = new Renderer[numNodes];
        nodeMats = new Material[numNodes];
        nodeActivities = new float[numNodes];

        for (int i = 0; i < numNodes; i++)
        {
            float angle = (float)i / numNodes * Mathf.PI * 2f;
            float ring = (i % 2 == 0) ? networkRadius : networkRadius * 0.6f;
            Vector3 pos = new Vector3(
                Mathf.Cos(angle) * ring,
                Mathf.Sin(angle * 0.5f) * 0.2f,
                Mathf.Sin(angle) * ring
            );

            var node = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            node.name = $"Node_{i}";
            node.transform.SetParent(transform);
            node.transform.localPosition = pos;
            node.transform.localScale = Vector3.one * nodeSize;

            // Remove collider for performance
            Destroy(node.GetComponent<Collider>());

            nodeMats[i] = new Material(Shader.Find("Standard"));
            nodeMats[i].EnableKeyword("_EMISSION");
            nodeMats[i].color = nodeInactive;
            node.GetComponent<Renderer>().material = nodeMats[i];

            nodes[i] = node;
            nodeRenderers[i] = node.GetComponent<Renderer>();
        }

        // Create sparse connections
        Random.InitState(42);
        connectionMask = new bool[numNodes, numNodes];
        var connectionList = new List<LineRenderer>();

        for (int i = 0; i < numNodes; i++)
        {
            for (int j = i + 1; j < numNodes; j++)
            {
                if (Random.value > sparsity)
                {
                    connectionMask[i, j] = true;
                    connectionMask[j, i] = true;

                    var connObj = new GameObject($"Conn_{i}_{j}");
                    connObj.transform.SetParent(transform);

                    var lr = connObj.AddComponent<LineRenderer>();
                    lr.positionCount = 2;
                    lr.startWidth = connectionWidth;
                    lr.endWidth = connectionWidth;

                    lr.material = new Material(Shader.Find("Sprites/Default"));
                    lr.startColor = connectionColor;
                    lr.endColor = connectionColor;
                    lr.useWorldSpace = false;

                    lr.SetPosition(0, nodes[i].transform.localPosition);
                    lr.SetPosition(1, nodes[j].transform.localPosition);

                    connectionList.Add(lr);
                }
            }
        }

        connections = connectionList.ToArray();
    }

    /// <summary>
    /// Set overall network activity level (0-1). Drives node and connection glow.
    /// </summary>
    public void SetActivity(float activity)
    {
        globalActivity = activity;
    }

    void Update()
    {
        if (connectionMask == null || nodes == null) return;
        float time = Time.time;

        // Propagate activity through nodes with wave pattern
        for (int i = 0; i < numNodes; i++)
        {
            float phase = (float)i / numNodes * Mathf.PI * 2f;
            float wave = Mathf.Sin(time * pulseSpeed + phase) * 0.5f + 0.5f;
            float targetActivity = globalActivity * wave;

            nodeActivities[i] = Mathf.Lerp(nodeActivities[i], targetActivity, Time.deltaTime * activityDecay);

            Color c = Color.Lerp(nodeInactive, nodeActive, nodeActivities[i]);
            nodeMats[i].color = c;
            nodeMats[i].SetColor("_EmissionColor", nodeActive * nodeActivities[i] * 2f);

            // Pulse node size
            float scale = nodeSize * (1f + nodeActivities[i] * 0.5f);
            nodes[i].transform.localScale = Vector3.one * scale;
        }

        // Animate connections
        int connIdx = 0;
        for (int i = 0; i < numNodes && connIdx < connections.Length; i++)
        {
            for (int j = i + 1; j < numNodes && connIdx < connections.Length; j++)
            {
                if (connectionMask[i, j])
                {
                    float avgActivity = (nodeActivities[i] + nodeActivities[j]) * 0.5f;
                    Color c = Color.Lerp(connectionColor, connectionActive, avgActivity);
                    connections[connIdx].startColor = c;
                    connections[connIdx].endColor = c;
                    connections[connIdx].startWidth = connectionWidth * (1f + avgActivity * 2f);
                    connections[connIdx].endWidth = connectionWidth * (1f + avgActivity * 2f);
                    connIdx++;
                }
            }
        }

        // Gentle rotation for visual appeal
        transform.Rotate(Vector3.up, 15f * Time.deltaTime, Space.Self);
    }
}
