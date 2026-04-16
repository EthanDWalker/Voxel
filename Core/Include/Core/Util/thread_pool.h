#pragma once

#include <condition_variable>
#include <functional>
#include <queue>
#include <thread>
#include <vector>

namespace Core {
struct ThreadPool {
  static std::vector<std::thread> threads;

  static std::queue<std::function<void(u32)>> all_thread_queue;
  static std::vector<u8> all_thread_task_complete;

  static std::queue<std::function<void(u32)>> task_queue;

  static std::mutex queue_mutex;
  static std::condition_variable cv;
  static bool stop;

  static void StartUp(const u32 thread_count = std::thread::hardware_concurrency() / 2 + 1) {
    ZoneScoped;
    all_thread_task_complete.resize(thread_count);
    for (u32 i = 0; i < thread_count; i++) {
      threads.emplace_back([i]() {
        std::function<void(u32)> task;

        while (true) {
          {
            std::unique_lock<std::mutex> lock(queue_mutex);

            cv.wait(lock, []() { return !task_queue.empty() || !all_thread_queue.empty() || stop; });

            if (stop && task_queue.empty() && all_thread_queue.empty()) {
              return;
            }

            if (!all_thread_queue.empty() && !all_thread_task_complete[i]) {
              task = all_thread_queue.front();
              all_thread_task_complete[i] = true;

              bool all_threads_done = true;
              for (u32 i = 0; i < all_thread_task_complete.size(); i++) {
                if (!all_thread_task_complete[i]) {
                  all_threads_done = false;
                  break;
                }
              }

              if (all_threads_done) {
                all_thread_queue.pop();
                for (u32 i = 0; i < all_thread_task_complete.size(); i++) {
                  all_thread_task_complete[i] = false;
                }
                cv.notify_all();
              }

              task(i);
            } else if (!task_queue.empty()) {
              task = std::move(task_queue.front());
              task_queue.pop();

              task(i);
            }
          }
        }
      });
    }
  }

  static void QueueTask(std::function<void(u32)> task) {
    ZoneScoped;
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      task_queue.emplace(task);
    }
    cv.notify_one();
  }

  static void CreateThreadLocalData(std::function<void(u32)> task) {
    ZoneScoped;
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      all_thread_queue.emplace(task);
    }
    cv.notify_all();
  }

  static void DestroyThreadLocalData(std::function<void(u32)> task) {
    ZoneScoped;
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      all_thread_queue.emplace(task);
    }

    cv.notify_all();
  }

  static void ShutDown() {
    ZoneScoped;
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stop = true;
    }
    cv.notify_all();

    for (auto &thread : threads) {
      thread.join();
    }
  }
};
} // namespace Core
