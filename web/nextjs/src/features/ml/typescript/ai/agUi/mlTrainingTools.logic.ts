/**
 * Resolves common ML form patch args from Copilot tool payloads.
 *
 * Supports both payload styles:
 * - `{ fields: { ...patch } }`
 * - `{ ...patch }`
 */
export function resolveMlFormPatchFromToolArgs<TPatch extends Record<string, unknown>>(
  args: Record<string, unknown>
): TPatch {
  const patchCandidate = args.fields;
  const patch =
    patchCandidate && typeof patchCandidate === "object" && !Array.isArray(patchCandidate)
      ? { ...(patchCandidate as Record<string, unknown>) }
      : { ...args };

  if (patch.set_sweep_values !== undefined && patch.run_sweep === undefined) {
    patch.run_sweep = patch.set_sweep_values;
  }

  return patch as TPatch;
}
