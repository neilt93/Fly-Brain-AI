using UnityEngine;

/// <summary>
/// Runtime binary STL parser. Converts MuJoCo-frame meshes to Unity coordinates.
/// </summary>
public static class StlImporter
{
    /// <summary>
    /// Parse binary STL data and return a Unity Mesh.
    /// mjScale is the MJCF mesh scale (e.g. 1000,1000,1000 or 1000,-1000,1000 for mirrored).
    /// Coordinate conversion: MuJoCo (X=fwd,Y=left,Z=up) → Unity (X=right,Y=up,Z=fwd).
    /// </summary>
    public static Mesh Load(byte[] data, Vector3 mjScale)
    {
        if (data == null || data.Length < 84)
        {
            Debug.LogError("StlImporter: invalid data");
            return null;
        }

        // 80-byte header (skip), then uint32 triangle count
        uint triCount = System.BitConverter.ToUInt32(data, 80);
        int expected = 84 + (int)triCount * 50;
        if (data.Length < expected)
        {
            Debug.LogError($"StlImporter: file too short for {triCount} triangles");
            return null;
        }

        var vertices = new Vector3[triCount * 3];
        var normals = new Vector3[triCount * 3];
        var triangles = new int[triCount * 3];

        // Determine if we need to flip winding (negative determinant = mirrored)
        bool flipWinding = (mjScale.x * mjScale.y * mjScale.z) < 0;

        int offset = 84;
        for (int i = 0; i < (int)triCount; i++)
        {
            // Normal (3 floats)
            float nx = System.BitConverter.ToSingle(data, offset);
            float ny = System.BitConverter.ToSingle(data, offset + 4);
            float nz = System.BitConverter.ToSingle(data, offset + 8);
            offset += 12;

            // Convert normal: MuJoCo → Unity (sign only, normals are unit vectors)
            Vector3 normal = new Vector3(
                -ny * Mathf.Sign(mjScale.y),
                 nz * Mathf.Sign(mjScale.z),
                 nx * Mathf.Sign(mjScale.x)
            ).normalized;

            // 3 vertices (each 3 floats)
            for (int v = 0; v < 3; v++)
            {
                float rx = System.BitConverter.ToSingle(data, offset);
                float ry = System.BitConverter.ToSingle(data, offset + 4);
                float rz = System.BitConverter.ToSingle(data, offset + 8);
                offset += 12;

                // Apply MJCF scale and convert coords
                vertices[i * 3 + v] = new Vector3(
                    -ry * mjScale.y,
                     rz * mjScale.z,
                     rx * mjScale.x
                );
                normals[i * 3 + v] = normal;
            }

            // 2-byte attribute (skip)
            offset += 2;

            // Triangle indices (flip winding for mirrored meshes)
            int baseIdx = i * 3;
            if (flipWinding)
            {
                triangles[baseIdx] = baseIdx;
                triangles[baseIdx + 1] = baseIdx + 2;
                triangles[baseIdx + 2] = baseIdx + 1;
            }
            else
            {
                triangles[baseIdx] = baseIdx;
                triangles[baseIdx + 1] = baseIdx + 1;
                triangles[baseIdx + 2] = baseIdx + 2;
            }
        }

        var mesh = new Mesh();
        if (vertices.Length > 65535)
            mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        mesh.vertices = vertices;
        mesh.normals = normals;
        mesh.triangles = triangles;
        mesh.RecalculateBounds();
        return mesh;
    }
}
