#pragma once

#include <condition_variable>
#include <functional>
#include <queue>
#include <thread>
#include <tracy/Tracy.hpp>
#include <vector>

namespace Core {
struct ThreadPool {
  static std::vector<std::thread> threads;

  static std::queue<std::function<void()>> all_thread_queue;
  static std::vector<u8> all_thread_task_complete;

  static std::queue<std::function<void()>> task_queue;

  static std::mutex queue_mutex;
  static std::condition_variable work_cv;
  static std::condition_variable completion_cv;
  static bool stop;

  static void StartUp(const u32 thread_count = std::thread::hardware_concurrency() / 2 + 1) {
    ZoneScoped;
    all_thread_task_complete.resize(thread_count);
    for (u32 i = 0; i < thread_count; i++) {
      threads.emplace_back([i]() {
        std::function<void()> task;

        while (true) {
          bool notify = false;
          {
            std::unique_lock<std::mutex> lock(queue_mutex);

            work_cv.wait(lock, []() { return !task_queue.empty() || !all_thread_queue.empty() || stop; });

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

                notify = true;
              }

              lock.unlock();
              task();
            } else if (!task_queue.empty()) {
              task = std::move(task_queue.front());
              task_queue.pop();

              lock.unlock();
              task();
            }
          }

          if (notify) {
            completion_cv.notify_all();
          }
        }
      });
    }
  }

  static void QueueTask(std::function<void()> task) {
    ZoneScoped;
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      task_queue.emplace(task);
    }
    work_cv.notify_one();
  }

  static void WaitForThreadLocalData() {
    ZoneScoped;

    std::unique_lock<std::mutex> lock(queue_mutex);
    completion_cv.wait(lock, []() { return all_thread_queue.empty(); });
  }

  static void CreateThreadLocalData(std::function<void()> task) {
    ZoneScoped;
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      all_thread_queue.emplace(task);
    }
    work_cv.notify_all();
  }

  static void DestroyThreadLocalData(std::function<void()> task) {
    ZoneScoped;
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      all_thread_queue.emplace(task);
    }

    work_cv.notify_all();
  }

  static void ShutDown() {
    ZoneScoped;
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stop = true;
    }
    work_cv.notify_all();

    for (auto &thread : threads) {
      thread.join();
    }
  }
};
} // namespace Core
