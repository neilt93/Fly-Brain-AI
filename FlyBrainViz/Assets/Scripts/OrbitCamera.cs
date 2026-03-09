using UnityEngine;

/// <summary>
/// Mouse-controlled orbit camera using legacy Input API for maximum compatibility.
/// Left-drag or Right-drag: orbit. Scroll: zoom. Middle-drag: pan. F: recenter.
/// </summary>
public class OrbitCamera : MonoBehaviour
{
    public Transform target;
    public float distance = 4f;
    public float orbitSpeed = 3f;
    public float zoomSpeed = 0.5f;
    public float panSpeed = 0.01f;
    public float minDistance = 0.5f;
    public float maxDistance = 20f;

    float yaw = 30f;
    float pitch = 20f;
    Vector3 focusOffset;

    void LateUpdate()
    {
        float mx = Input.GetAxis("Mouse X");
        float my = Input.GetAxis("Mouse Y");

        // Left-click or right-click orbit
        if (Input.GetMouseButton(0) || Input.GetMouseButton(1))
        {
            yaw += mx * orbitSpeed;
            pitch -= my * orbitSpeed;
            pitch = Mathf.Clamp(pitch, -80f, 80f);
        }

        // Middle-click pan
        if (Input.GetMouseButton(2))
        {
            focusOffset -= transform.right * mx * panSpeed;
            focusOffset -= transform.up * my * panSpeed;
        }

        // Scroll zoom
        float scroll = Input.mouseScrollDelta.y;
        if (scroll != 0f)
            distance = Mathf.Clamp(distance - scroll * zoomSpeed, minDistance, maxDistance);

        // F key: recenter
        if (Input.GetKeyDown(KeyCode.F))
            focusOffset = Vector3.zero;

        // Apply
        Vector3 focus = (target != null ? target.position : Vector3.zero) + focusOffset;
        Quaternion rot = Quaternion.Euler(pitch, yaw, 0f);
        transform.position = focus - rot * Vector3.forward * distance;
        transform.rotation = rot;
    }
}
