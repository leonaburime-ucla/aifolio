/**
 * Copilot message persistence orchestration decisions.
 *
 * Purpose:
 * - Keep hydration/sync branching deterministic and testable.
 * - Prevent transient runtime snapshots from clobbering persisted history.
 */

/**
 * Returns whether persisted history should hydrate live runtime state.
 *
 * @param livePersistableCount Count of persistable live messages.
 * @param liveUserMessageCount Count of live user-authored messages.
 * @param persistedCount Count of persisted messages.
 * @returns `true` when persisted history should be restored into live state.
 */
export function shouldHydratePersistedMessages({
  livePersistableCount,
  liveUserMessageCount,
  persistedCount,
}: {
  livePersistableCount: number;
  liveUserMessageCount: number;
  persistedCount: number;
}): boolean {
  if (persistedCount === 0) return false;
  // Hydrate when runtime has no user-authored messages yet.
  // Copilot may mount transient assistant/system snapshots before restore.
  if (liveUserMessageCount === 0) return true;
  return livePersistableCount === 0;
}

/**
 * Returns whether sync should skip writing empty persistable snapshots.
 *
 * @param livePersistableCount Count of persistable live messages.
 * @param persistedCount Count of persisted messages.
 * @returns `true` when empty live snapshot should not overwrite non-empty persisted history.
 */
export function shouldSkipEmptyPersistableSync({
  livePersistableCount,
  persistedCount,
}: {
  livePersistableCount: number;
  persistedCount: number;
}): boolean {
  return livePersistableCount === 0 && persistedCount > 0;
}
