using UnityEngine;
using System.Collections.Generic;
using System.Xml;
using System.IO;

/// <summary>
/// Parses the NeuroMechFly MJCF XML and builds the anatomical fly hierarchy at runtime.
/// Loads STL meshes via StlImporter, assigns appearance colors, and creates JointInfo components.
/// </summary>
public static class MjcfFlyBuilder
{
    public class BuildResult
    {
        public GameObject root;
        public List<JointInfo> animatedBodies = new List<JointInfo>();
        public Dictionary<string, Transform> bodyTransforms = new Dictionary<string, Transform>();
        public Dictionary<string, List<Renderer>> legRenderers = new Dictionary<string, List<Renderer>>();
        public Renderer thoraxRenderer;
        public List<Renderer> abdomenRenderers = new List<Renderer>();
    }

    struct MeshAsset
    {
        public string stlBaseName; // e.g. "RFCoxa" (without .stl)
        public Vector3 scale;      // MJCF scale, e.g. (1000, -1000, 1000)
    }

    // ── Appearance colors from flygym config.yaml ──
    static readonly Dictionary<string, Color> BodyColors = new Dictionary<string, Color>
    {
        // Eyes
        { "LEye",  new Color(0.67f, 0.21f, 0.12f) },
        { "REye",  new Color(0.67f, 0.21f, 0.12f) },
        // Head & Thorax
        { "Head",       new Color(0.59f, 0.39f, 0.12f) },
        { "Thorax",     new Color(0.59f, 0.39f, 0.12f) },
        // Proboscis
        { "Rostrum",    new Color(0.59f, 0.39f, 0.12f) },
        { "Haustellum", new Color(0.59f, 0.39f, 0.12f) },
        // Abdomen gradient
        { "A1A2", new Color(0.59f, 0.39f, 0.12f) },
        { "A3",   new Color(0.64f, 0.46f, 0.20f) },
        { "A4",   new Color(0.70f, 0.53f, 0.28f) },
        { "A5",   new Color(0.76f, 0.60f, 0.35f) },
        { "A6",   new Color(0.39f, 0.20f, 0.00f) },
        // Antenna parts
        { "LPedicel",  new Color(0.59f, 0.39f, 0.12f) },
        { "RPedicel",  new Color(0.59f, 0.39f, 0.12f) },
        { "LFuniculus", new Color(0.59f, 0.39f, 0.12f) },
        { "RFuniculus", new Color(0.59f, 0.39f, 0.12f) },
        { "LArista",   new Color(0.26f, 0.20f, 0.16f) },
        { "RArista",   new Color(0.26f, 0.20f, 0.16f) },
        // Halteres
        { "LHaltere", new Color(0.59f, 0.43f, 0.24f) },
        { "RHaltere", new Color(0.59f, 0.43f, 0.24f) },
    };

    static readonly string[] WingNames = { "LWing", "RWing" };
    static readonly Color WingColor = new Color(0.8f, 0.8f, 0.9f, 0.3f);

    // Leg segment colors
    static readonly Color CoxaColor  = new Color(0.59f, 0.39f, 0.12f);
    static readonly Color FemurColor = new Color(0.63f, 0.43f, 0.16f);
    static readonly Color TibiaColor = new Color(0.67f, 0.47f, 0.20f);
    static readonly Color TarsusColor = new Color(0.71f, 0.51f, 0.24f);

    static readonly string[] LegPrefixes = { "LF", "LM", "LH", "RF", "RM", "RH" };

    public static BuildResult BuildFly(Transform parent, float globalScale = 0.5f)
    {
        var result = new BuildResult();

        // Initialize leg renderer lists
        foreach (var prefix in LegPrefixes)
            result.legRenderers[prefix] = new List<Renderer>();

        // Load MJCF XML from Resources
        TextAsset xmlAsset = Resources.Load<TextAsset>("neuromechfly");
        if (xmlAsset == null)
        {
            Debug.LogError("MjcfFlyBuilder: neuromechfly.xml not found in Resources!");
            return result;
        }

        var doc = new XmlDocument();
        doc.LoadXml(xmlAsset.text);

        // 1. Parse <asset> → mesh definitions
        var meshAssets = new Dictionary<string, MeshAsset>();
        var assetNode = doc.SelectSingleNode("//asset");
        if (assetNode != null)
        {
            foreach (XmlNode meshNode in assetNode.SelectNodes("mesh"))
            {
                string meshName = meshNode.Attributes["name"]?.Value;
                string filePath = meshNode.Attributes["file"]?.Value;
                string scaleStr = meshNode.Attributes["scale"]?.Value;
                if (meshName == null || filePath == null) continue;

                // Extract base name from path (e.g. "../mesh/RFCoxa.stl" → "RFCoxa")
                string baseName = Path.GetFileNameWithoutExtension(filePath);

                Vector3 scale = new Vector3(1000, 1000, 1000);
                if (scaleStr != null)
                {
                    string[] parts = scaleStr.Trim().Split(' ');
                    if (parts.Length == 3)
                    {
                        float.TryParse(parts[0], System.Globalization.NumberStyles.Float,
                            System.Globalization.CultureInfo.InvariantCulture, out scale.x);
                        float.TryParse(parts[1], System.Globalization.NumberStyles.Float,
                            System.Globalization.CultureInfo.InvariantCulture, out scale.y);
                        float.TryParse(parts[2], System.Globalization.NumberStyles.Float,
                            System.Globalization.CultureInfo.InvariantCulture, out scale.z);
                    }
                }

                meshAssets[meshName] = new MeshAsset { stlBaseName = baseName, scale = scale };
            }
        }

        // 2. Find the root body (Thorax under FlyBody)
        XmlNode thoraxNode = doc.SelectSingleNode("//worldbody/body[@name='FlyBody']/body[@name='Thorax']");
        if (thoraxNode == null)
        {
            Debug.LogError("MjcfFlyBuilder: Thorax body not found in MJCF!");
            return result;
        }

        // Create root GameObject
        var rootObj = new GameObject("FlyRoot");
        rootObj.transform.SetParent(parent, false);
        rootObj.transform.localScale = Vector3.one * globalScale;
        result.root = rootObj;

        // 3. Recursively build hierarchy starting from Thorax
        BuildBody(thoraxNode, rootObj.transform, meshAssets, result);

        // Zero out Thorax local position — experiment data already provides
        // the Thorax world position, so we don't want the MJCF offset from FlyBody
        if (result.bodyTransforms.TryGetValue("Thorax", out var thoraxTransform))
            thoraxTransform.localPosition = Vector3.zero;

        return result;
    }

    static void BuildBody(XmlNode bodyNode, Transform parentTransform,
        Dictionary<string, MeshAsset> meshAssets, BuildResult result)
    {
        string bodyName = bodyNode.Attributes["name"]?.Value ?? "unnamed";

        // Create GameObject
        var obj = new GameObject(bodyName);
        obj.transform.SetParent(parentTransform, false);

        // Set local position from MJCF pos (MuJoCo→Unity)
        string posStr = bodyNode.Attributes["pos"]?.Value;
        if (posStr != null)
            obj.transform.localPosition = ParseMjPosition(posStr);

        // Set local rotation from MJCF quat (MuJoCo→Unity)
        string quatStr = bodyNode.Attributes["quat"]?.Value;
        Quaternion baseRot = Quaternion.identity;
        if (quatStr != null)
        {
            baseRot = ParseMjQuaternion(quatStr);
            obj.transform.localRotation = baseRot;
        }

        result.bodyTransforms[bodyName] = obj.transform;

        // Process <geom> elements — attach meshes
        foreach (XmlNode geomNode in bodyNode.SelectNodes("geom"))
        {
            string meshAttr = geomNode.Attributes["mesh"]?.Value;
            if (meshAttr == null || !meshAssets.ContainsKey(meshAttr)) continue;

            var asset = meshAssets[meshAttr];
            Mesh mesh = LoadStlMesh(asset.stlBaseName, asset.scale);
            if (mesh == null) continue;

            // Create a child for the mesh (geoms can have their own pos/quat)
            string geomPos = geomNode.Attributes["pos"]?.Value;
            string geomQuat = geomNode.Attributes["quat"]?.Value;

            GameObject meshObj;
            if (geomPos != null || geomQuat != null)
            {
                meshObj = new GameObject(bodyName + "_mesh");
                meshObj.transform.SetParent(obj.transform, false);
                if (geomPos != null)
                    meshObj.transform.localPosition = ParseMjPosition(geomPos);
                if (geomQuat != null)
                    meshObj.transform.localRotation = ParseMjQuaternion(geomQuat);
            }
            else
            {
                meshObj = obj;
            }

            // Add mesh components (may already have them if meshObj == obj)
            var mf = meshObj.GetComponent<MeshFilter>();
            if (mf == null) mf = meshObj.AddComponent<MeshFilter>();
            mf.mesh = mesh;

            var mr = meshObj.GetComponent<MeshRenderer>();
            if (mr == null) mr = meshObj.AddComponent<MeshRenderer>();

            // Assign material with appearance color
            mr.material = CreateMaterial(bodyName);

            // Track renderers for runtime coloring
            CacheRenderer(bodyName, mr, result);
        }

        // Process <joint> elements — create JointInfo
        var jointNodes = bodyNode.SelectNodes("joint");
        if (jointNodes.Count > 0)
        {
            var ji = obj.AddComponent<JointInfo>();
            ji.baseRotation = baseRot;
            var names = new List<string>();
            var axes = new List<Vector3>();

            foreach (XmlNode jointNode in jointNodes)
            {
                string jName = jointNode.Attributes["name"]?.Value;
                string axisStr = jointNode.Attributes["axis"]?.Value;
                if (jName == null || axisStr == null) continue;

                names.Add(jName);
                axes.Add(ParseMjAxis(axisStr));
            }

            ji.jointNames = names.ToArray();
            ji.jointAxes = axes.ToArray();
            result.animatedBodies.Add(ji);
        }

        // Recurse into child <body> elements
        foreach (XmlNode child in bodyNode.SelectNodes("body"))
        {
            BuildBody(child, obj.transform, meshAssets, result);
        }
    }

    static void CacheRenderer(string bodyName, Renderer r, BuildResult result)
    {
        if (bodyName == "Thorax")
        {
            result.thoraxRenderer = r;
            return;
        }

        if (bodyName == "A1A2" || bodyName == "A3" || bodyName == "A4" ||
            bodyName == "A5" || bodyName == "A6")
        {
            result.abdomenRenderers.Add(r);
            return;
        }

        // Check if it's a leg segment
        foreach (var prefix in LegPrefixes)
        {
            if (bodyName.StartsWith(prefix))
            {
                result.legRenderers[prefix].Add(r);
                return;
            }
        }
    }

    // ── Coordinate conversion helpers ──

    static Vector3 ParseMjPosition(string posStr)
    {
        string[] parts = posStr.Trim().Split(new[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length < 3) return Vector3.zero;

        float mx = ParseFloat(parts[0]);
        float my = ParseFloat(parts[1]);
        float mz = ParseFloat(parts[2]);

        // MuJoCo (X=fwd,Y=left,Z=up) → Unity (X=right,Y=up,Z=fwd)
        return new Vector3(-my, mz, mx);
    }

    static Quaternion ParseMjQuaternion(string quatStr)
    {
        string[] parts = quatStr.Trim().Split(new[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length < 4) return Quaternion.identity;

        // MJCF quat order: w, x, y, z (MuJoCo frame)
        float mw = ParseFloat(parts[0]);
        float mqx = ParseFloat(parts[1]);
        float mqy = ParseFloat(parts[2]);
        float mqz = ParseFloat(parts[3]);

        // MuJoCo → Unity quaternion (imaginary parts are pseudovectors, det=-1)
        return new Quaternion(mqy, -mqz, -mqx, mw);
    }

    static Vector3 ParseMjAxis(string axisStr)
    {
        string[] parts = axisStr.Trim().Split(new[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length < 3) return Vector3.up;

        float ax = ParseFloat(parts[0]);
        float ay = ParseFloat(parts[1]);
        float az = ParseFloat(parts[2]);

        // Pseudovector conversion (det=-1 transform): multiply by -M
        return new Vector3(ay, -az, -ax);
    }

    static float ParseFloat(string s)
    {
        float.TryParse(s, System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture, out float v);
        return v;
    }

    // ── STL loading ──

    static Dictionary<string, Mesh> meshCache = new Dictionary<string, Mesh>();

    static Mesh LoadStlMesh(string baseName, Vector3 mjScale)
    {
        // Cache key includes scale to handle L/R variants of same file
        string key = $"{baseName}_{mjScale.x}_{mjScale.y}_{mjScale.z}";
        if (meshCache.TryGetValue(key, out Mesh cached))
            return cached;

        byte[] data = LoadStlBytes(baseName);
        if (data == null) return null;

        Mesh mesh = StlImporter.Load(data, mjScale);
        if (mesh != null)
        {
            mesh.name = baseName;
            meshCache[key] = mesh;
        }
        return mesh;
    }

    static byte[] LoadStlBytes(string baseName)
    {
        // Try Resources.Load first (works if Unity imports .stl as TextAsset)
        var ta = Resources.Load<TextAsset>("FlyMesh/" + baseName);
        if (ta != null) return ta.bytes;

        // Fall back to file system (works in editor and standalone with files present)
        string path = Path.Combine(Application.dataPath, "Resources", "FlyMesh", baseName + ".stl");
        if (File.Exists(path)) return File.ReadAllBytes(path);

        Debug.LogWarning($"MjcfFlyBuilder: STL not found: {baseName}");
        return null;
    }

    // ── Material creation ──

    static Material CreateMaterial(string bodyName)
    {
        // Wings — transparent
        foreach (var wn in WingNames)
        {
            if (bodyName == wn)
                return CreateTransparentMaterial(WingColor);
        }

        // Direct color lookup
        if (BodyColors.TryGetValue(bodyName, out Color c))
            return CreateStandardMaterial(c);

        // Leg segment colors by suffix
        foreach (var prefix in LegPrefixes)
        {
            if (!bodyName.StartsWith(prefix)) continue;
            string segment = bodyName.Substring(prefix.Length);
            if (segment.StartsWith("Coxa"))  return CreateStandardMaterial(CoxaColor);
            if (segment.StartsWith("Femur")) return CreateStandardMaterial(FemurColor);
            if (segment.StartsWith("Tibia")) return CreateStandardMaterial(TibiaColor);
            if (segment.StartsWith("Tarsus")) return CreateStandardMaterial(TarsusColor);
        }

        // Default body color
        return CreateStandardMaterial(new Color(0.59f, 0.39f, 0.12f));
    }

    static Material CreateStandardMaterial(Color color)
    {
        var mat = new Material(Shader.Find("Standard"));
        mat.color = color;
        mat.EnableKeyword("_EMISSION");
        mat.SetColor("_EmissionColor", Color.black);
        return mat;
    }

    static Material CreateTransparentMaterial(Color color)
    {
        var mat = new Material(Shader.Find("Standard"));
        mat.color = color;
        mat.SetFloat("_Mode", 3); // Transparent
        mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        mat.SetInt("_ZWrite", 0);
        mat.DisableKeyword("_ALPHATEST_ON");
        mat.EnableKeyword("_ALPHABLEND_ON");
        mat.DisableKeyword("_ALPHAPREMULTIPLY_ON");
        mat.renderQueue = 3000;
        return mat;
    }
}
