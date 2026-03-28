#include "Core/Util/thread_pool.h"

namespace Core {
std::vector<std::thread> ThreadPool::threads{};
std::queue<std::function<void(u32)>> ThreadPool::task_queue{};

std::queue<std::function<void(u32)>> ThreadPool::all_thread_queue{};
std::vector<u8> ThreadPool::all_thread_task_complete;

std::mutex ThreadPool::queue_mutex{};
std::condition_variable ThreadPool::cv{};
bool ThreadPool::stop = false;
} // namespace Core
