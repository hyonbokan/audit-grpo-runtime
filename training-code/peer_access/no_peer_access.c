/* Preloaded on hosts whose GPU peer copies are corrupt (host.env AUDIT_GRPO_PEER_ACCESS=disabled).
 * Reporting no peer access makes CUDA stage every copy between GPUs through host memory. */
int cudaDeviceCanAccessPeer(int *can_access, int device, int peer) {
  (void)device;
  (void)peer;
  *can_access = 0;
  return 0; /* cudaSuccess */
}

int cuDeviceCanAccessPeer(int *can_access, int device, int peer) {
  (void)device;
  (void)peer;
  *can_access = 0;
  return 0; /* CUDA_SUCCESS */
}
