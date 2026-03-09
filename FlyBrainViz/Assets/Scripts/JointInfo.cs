using UnityEngine;

/// <summary>
/// Stores joint metadata for an MJCF body that has one or more hinge joints.
/// Multiple joints on the same body are composed in XML order.
/// </summary>
public class JointInfo : MonoBehaviour
{
    public string[] jointNames;
    public Vector3[] jointAxes; // already in Unity coordinates
    public Quaternion baseRotation;

    /// <summary>
    /// Apply composed rotation from current angle values.
    /// angles maps joint name → radians via the caller.
    /// </summary>
    public void ApplyAngles(System.Collections.Generic.Dictionary<string, float> angles)
    {
        Quaternion rot = baseRotation;
        for (int i = 0; i < jointNames.Length; i++)
        {
            if (angles.TryGetValue(jointNames[i], out float rad))
            {
                rot *= Quaternion.AngleAxis(rad * Mathf.Rad2Deg, jointAxes[i]);
            }
        }
        transform.localRotation = rot;
    }
}
