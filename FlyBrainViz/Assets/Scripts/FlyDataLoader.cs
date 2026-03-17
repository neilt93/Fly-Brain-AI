using UnityEngine;
using System.Collections.Generic;
using Newtonsoft.Json;

/// <summary>
/// Loads experiment time-series JSON and connectome activity data from Resources.
/// Provides frame-by-frame fly position, contact states, joint angles, tripod scores,
/// and connectome neural firing rates.
/// </summary>
public class FlyDataLoader : MonoBehaviour
{
    public class TimeSeriesData
    {
        public string controller;
        public float dt;
        public int n_frames;
        public List<List<float>> positions;          // [frame][3]
        public List<List<float>> contacts;           // [frame][6] — floats in JSON (1.0/0.0)
        public List<List<float>> contact_forces;     // [frame][6]
        public List<List<List<float>>> end_effectors; // [frame][6][3]
        public List<List<float>> joint_angles;       // [frame][42]
        public List<string> joint_names;             // [42]
        public List<float> tripod_score;
        public List<float> weight_drifts;
        public int? perturbation_idx;
    }

    /// <summary>
    /// Connectome activity data from the LIF brain simulation.
    /// Matches the JSON schema exported by export_connectome_viz.py.
    /// </summary>
    public class ConnectomeData
    {
        public int n_neurons;
        public int n_frames;
        public int n_total_connectome;
        public List<string> neuron_types;
        public List<string> neuron_names;
        public List<List<float>> positions_3d;         // [neuron][3]
        public List<List<float>> connections;           // [conn][3]: from_idx, to_idx, weight
        public List<List<float>> firing_rates;          // [frame][neuron]
        public Dictionary<string, object> metadata;
    }

    public TimeSeriesData plasticData;
    public TimeSeriesData fixedData;
    public ConnectomeData connectomeData;
    public bool dataLoaded = false;
    public bool connectomeLoaded = false;

    void Awake()
    {
        LoadData();
    }

    public void LoadData()
    {
        TextAsset plasticJson = Resources.Load<TextAsset>("timeseries_plastic");
        TextAsset fixedJson = Resources.Load<TextAsset>("timeseries_fixed");

        if (plasticJson != null)
        {
            try
            {
                plasticData = JsonConvert.DeserializeObject<TimeSeriesData>(plasticJson.text);
                string dofLabel = ", no joint angles";
                if (plasticData.joint_angles != null && plasticData.joint_angles.Count > 0)
                    dofLabel = $", {plasticData.joint_angles[0].Count} DOFs";
                Debug.Log($"Loaded plastic: {plasticData.n_frames} frames" +
                    dofLabel);
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Failed to parse timeseries_plastic.json: {e.Message}");
            }
        }
        else
        {
            Debug.LogError("Missing timeseries_plastic.json in Resources!");
        }

        if (fixedJson != null)
        {
            try
            {
                fixedData = JsonConvert.DeserializeObject<TimeSeriesData>(fixedJson.text);
                Debug.Log($"Loaded fixed: {fixedData.n_frames} frames");
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Failed to parse timeseries_fixed.json: {e.Message}");
            }
        }

        // Load connectome activity data
        TextAsset connectomeJson = Resources.Load<TextAsset>("connectome_activity");
        if (connectomeJson != null)
        {
            try
            {
                connectomeData = JsonConvert.DeserializeObject<ConnectomeData>(connectomeJson.text);
                connectomeLoaded = true;
                Debug.Log($"Loaded connectome: {connectomeData.n_neurons} neurons, " +
                    $"{connectomeData.n_frames} frames, " +
                    $"{connectomeData.connections?.Count ?? 0} connections");
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Failed to parse connectome_activity.json: {e.Message}");
            }
        }
        else
        {
            Debug.LogWarning("No connectome_activity.json in Resources (connectome demo disabled)");
        }

        dataLoaded = plasticData != null;
    }

    /// <summary>
    /// Get firing rates for a specific frame from connectome data.
    /// Returns null if connectome data not loaded or frame out of range.
    /// </summary>
    public float[] GetFiringRates(int frame)
    {
        if (connectomeData == null || connectomeData.firing_rates == null || connectomeData.n_frames <= 0)
            return null;
        int idx = frame % connectomeData.n_frames;
        var rates = connectomeData.firing_rates[idx];
        float[] result = new float[rates.Count];
        for (int i = 0; i < rates.Count; i++)
            result[i] = rates[i];
        return result;
    }

    /// <summary>
    /// Get fly position in Unity coords. Positions stay in mm — the fly's globalScale handles sizing.
    /// MuJoCo (X=fwd, Y=left, Z=up) → Unity (X=right, Y=up, Z=fwd).
    /// </summary>
    public Vector3 GetPosition(TimeSeriesData data, int frame)
    {
        if (data == null || data.positions == null || frame >= data.positions.Count)
            return Vector3.zero;
        var p = data.positions[frame];
        if (p.Count < 3) return Vector3.zero;
        return new Vector3(-p[1], p[2], p[0]);
    }

    public int[] GetContacts(TimeSeriesData data, int frame)
    {
        if (data == null || data.contacts == null || frame >= data.contacts.Count)
            return new int[6];
        var floats = data.contacts[frame];
        var ints = new int[floats.Count];
        for (int i = 0; i < floats.Count; i++)
            ints[i] = floats[i] > 0.5f ? 1 : 0;
        return ints;
    }

    public float GetTripodScore(TimeSeriesData data, int frame)
    {
        if (data == null || data.tripod_score == null || frame >= data.tripod_score.Count)
            return 0f;
        return data.tripod_score[frame];
    }

    public float GetWeightDrift(TimeSeriesData data, int frame)
    {
        if (data == null || data.weight_drifts == null || data.weight_drifts.Count == 0)
            return 0f;
        int idx = Mathf.Min(frame, data.weight_drifts.Count - 1);
        return data.weight_drifts[idx];
    }

    /// <summary>
    /// Get joint angles for a frame (42 floats, radians).
    /// Returns null if joint angle data is not available.
    /// </summary>
    public float[] GetJointAngles(TimeSeriesData data, int frame)
    {
        if (data == null || data.joint_angles == null || frame >= data.joint_angles.Count)
            return null;
        return data.joint_angles[frame].ToArray();
    }

    /// <summary>
    /// Get joint names array (42 strings).
    /// Returns null if not available.
    /// </summary>
    public string[] GetJointNames(TimeSeriesData data)
    {
        if (data == null || data.joint_names == null || data.joint_names.Count == 0)
            return null;
        return data.joint_names.ToArray();
    }
}
