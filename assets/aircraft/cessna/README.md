# Optional aircraft model

The standalone replay viewer looks for a user-supplied model at:

```text
assets/aircraft/cessna/cessna.fbx
```

No license or redistribution terms accompanied the local development copy, so
the FBX is intentionally excluded from Git. Users must provide a model they are
legally permitted to use. The replay viewer automatically falls back to its
built-in primitive aircraft when the file is absent; CI and scientific tests do
not require the FBX.
