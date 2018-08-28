/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef LINGVO_CORE_OPS_MUTEX_H_
#define LINGVO_CORE_OPS_MUTEX_H_

#include <functional>

#include "nsync/public/nsync_mu.h"
#include "nsync/public/nsync_mu_wait.h"
#include "nsync/public/nsync_note.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace lingvo {

class Mutex;

class Condition {
 public:
  // Note that the function must be for the object passed - it cannot be a
  // superclass method.
  template <typename T>
  Condition(T* object, bool (T::*func)() const) {
    condition_ = [](const void* condition_arg) {
      auto fn = *static_cast<const std::function<bool()>*>(condition_arg);
      return (int)fn();
    };
    condition_arg_ = std::bind(func, object);
  }

 private:
  int (*condition_)(const void* condition_arg);
  std::function<bool()> condition_arg_;

  void Await(nsync::nsync_mu* mu) const {
    nsync::nsync_mu_wait(mu, condition_, &condition_arg_, nullptr);
  }

  friend class Mutex;

  TF_DISALLOW_COPY_AND_ASSIGN(Condition);
};

class Mutex {
 public:
  Mutex() : mu_(NSYNC_MU_INIT) {}

  void Lock() EXCLUSIVE_LOCK_FUNCTION() { nsync_mu_lock(&mu_); }

  void Unlock() UNLOCK_FUNCTION() { nsync_mu_unlock(&mu_); }

  void Await(const Condition& cond) {
    nsync_mu_rassert_held(&mu_);
    cond.Await(&mu_);
  }

 private:
  nsync::nsync_mu mu_;

  TF_DISALLOW_COPY_AND_ASSIGN(Mutex);
};

class SCOPED_LOCKABLE MutexLock {
 public:
  explicit MutexLock(Mutex* mu) EXCLUSIVE_LOCK_FUNCTION(mu) : mu_(mu) {
    mu_->Lock();
  }

  ~MutexLock() UNLOCK_FUNCTION() { mu_->Unlock(); }

 private:
  Mutex* const mu_;

  TF_DISALLOW_COPY_AND_ASSIGN(MutexLock);
};

class Notification {
 public:
  Notification()
      : note_(nsync::nsync_note_new(nullptr, nsync::nsync_time_no_deadline)) {}

  ~Notification() { nsync::nsync_note_free(note_); }

  void WaitForNotification() const {
    // TODO(zhifengc): Maybe optimize path the notification has already fired.
    nsync::nsync_note_wait(note_, nsync::nsync_time_no_deadline);
  }

  void Notify() { nsync::nsync_note_notify(note_); }

 private:
  nsync::nsync_note note_;

  TF_DISALLOW_COPY_AND_ASSIGN(Notification);
};

}  // namespace lingvo
}  // namespace tensorflow

#endif  // LINGVO_CORE_OPS_MUTEX_H_
