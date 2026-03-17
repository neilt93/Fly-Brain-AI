using UnityEngine;

/// <summary>
/// Mouse + keyboard orbit camera using legacy Input API for maximum compatibility.
/// Mouse: Left/Right-drag orbit, Scroll zoom, Middle-drag pan.
/// Keyboard: WASD move, QE up/down, Shift fast, F recenter.
/// </summary>
public class OrbitCamera : MonoBehaviour
{
    public Transform target;
    public float distance = 4f;
    public float orbitSpeed = 3f;
    public float zoomSpeed = 0.5f;
    public float panSpeed = 0.01f;
    public float moveSpeed = 3f;
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

        // WASD + QE keyboard movement
        float speed = moveSpeed * Time.deltaTime;
        if (Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift))
            speed *= 3f;

        Vector3 flatFwd = Vector3.ProjectOnPlane(transform.forward, Vector3.up).normalized;
        Vector3 flatRight = Vector3.ProjectOnPlane(transform.right, Vector3.up).normalized;

        if (Input.GetKey(KeyCode.W)) focusOffset += flatFwd * speed;
        if (Input.GetKey(KeyCode.S)) focusOffset -= flatFwd * speed;
        if (Input.GetKey(KeyCode.A)) focusOffset -= flatRight * speed;
        if (Input.GetKey(KeyCode.D)) focusOffset += flatRight * speed;
        if (Input.GetKey(KeyCode.E)) focusOffset += Vector3.up * speed;
        if (Input.GetKey(KeyCode.Q)) focusOffset -= Vector3.up * speed;

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
