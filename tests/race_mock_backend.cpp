#ifdef LEGACY_ONLY
extern "C" void *raceCreatePerception(const void*) { return nullptr; }
#else
extern "C" int raceBackendAbi() { return PROBE_ABI; }
extern "C" void *raceCreatePerceptionV2(const void*) { return nullptr; }
#endif
