// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <converse.h>

#include "Utilities/Gsl.hpp"
#include "Utilities/NoSuchType.hpp"

namespace Parallel {
/*!
 * \ingroup ParallelGroup
 * \brief A typesafe wrapper for the Converse nodelock, with safe creation,
 * destruction, and serialization.
 *
 * \details On construction, this class creates a Converse nodelock, and frees
 * the lock on destruction. During serialization, the contained nodelock at the
 * sending node is destroyed to avoid misuse, and throws an error if the lock is
 * used thereafter. When the lock is deserialized, the nodelock is re-created.
 */
class NodeLock {
 public:
  NodeLock() noexcept {
    lock_ = CmiCreateLock();
  }

  explicit NodeLock(CkMigrateMessage* /*message*/) noexcept {}

  NodeLock(const NodeLock&) = delete;
  NodeLock& operator=(const NodeLock&) = delete;
  NodeLock(NodeLock&& moved_lock) noexcept {
    moved_lock.destroy();
    lock_ = CmiCreateLock();
  }

  NodeLock& operator=(NodeLock&& moved_lock) noexcept {
    moved_lock.destroy();
    lock_ = CmiCreateLock();
    return *this;
  }

  ~NodeLock() noexcept { destroy(); }

  void lock() noexcept {
  if (UNLIKELY(destroyed_)) {
    ERROR("Trying to lock a destroyed lock");
  }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
    CmiLock(lock_);
#pragma GCC diagnostic pop
}

  bool try_lock() noexcept {
    if (UNLIKELY(destroyed_)) {
      ERROR("Trying to try_lock a destroyed lock");
    }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
    return CmiTryLock(lock_) == 0;
#pragma GCC diagnostic pop
  }

  void unlock() noexcept {
    if (UNLIKELY(destroyed_)) {
      ERROR("Trying to unlock a destroyed lock");
    }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
    CmiUnlock(lock_);
#pragma GCC diagnostic pop
  }

  void destroy() noexcept {
    if(destroyed_) {
      return;
    }
    destroyed_ = true;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
    CmiDestroyLock(lock_);
#pragma GCC diagnostic pop
  }

  void pup(PUP::er& p) noexcept {  // NOLINT
    if (p.isUnpacking()) {
      lock_ = CmiCreateLock();
    } else if (p.isPacking()) {
      destroy();
    }
  }

 private:
  bool destroyed_ = false;
  CmiNodeLock lock_{};  // NOLINT
};
}  // namespace Parallel
