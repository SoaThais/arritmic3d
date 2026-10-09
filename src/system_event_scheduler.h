#ifndef SYSTEM_EVENT_SCHEDULER_H
#define SYSTEM_EVENT_SCHEDULER_H

#include <queue>
#include <vector>
#include <cstddef>
#include <fstream>
#include <stdexcept>

enum class SystemEventType : unsigned char {
    NODE_EVENT = 0, 
    EXT_ACTIVATION,
    FILE_WRITE,
    OTHER,
    NO_EVENT,
    SIZE
};

struct SystemEvent {
    float event_time;
    SystemEventType type;
    unsigned char priority;

    bool operator<(const SystemEvent & other) const {
        return this->event_time > other.event_time;
    }
};

class SystemEventScheduler {

    private:

        std::priority_queue<SystemEvent> events;

    public:

        void Clear() {
            while(!events.empty())
                events.pop();
        }

        void Insert(float time, SystemEventType type, unsigned char priority = 1) {
            events.push(SystemEvent{time, type, priority});
        }

        bool IsEmpty() const {
            return events.empty();
        }

        const SystemEvent & GetFirst() const {
            return events.top();
        }

        void ExtractFirst() {
            events.pop();
        }

        // void SaveState(std::ofstream & f) const {

        //     // Número de eventos
        //     size_t n_system_events = events.size();

        //     f.write((char *) &n_system_events, sizeof(size_t));

        //     // Copia da fila para poder percorrê-la sem modificar a original
        //     auto queue_copy = events;

        //     while(!queue_copy.empty()) {
        //         const SystemEvent & ev = queue_copy.top();
        //         f.write((char *)&ev, sizeof(SystemEvent));
        //         queue_copy.pop();
        //     }
        // }

        // void LoadState(std::ifstream & f) {

        //     Clear();

        //     size_t n_system_events;
        //     f.read((char *)&n_system_events, sizeof(size_t));

        //     for(size_t i = 0; i < n_system_events; ++i) {
        //         SystemEvent ev;
        //         f.read((char *)&ev, sizeof(SystemEvent));
        //         events.push(ev);
        //     }
        // }

        void SaveState(std::ofstream & f) const {

            // Número de eventos
            size_t n_system_events = events.size();

            f.write((char *) &n_system_events, sizeof(size_t));

            // Copia da fila para poder percorrê-la sem modificar a original
            auto queue_copy = events;

            while(!queue_copy.empty()) {
                const SystemEvent & ev = queue_copy.top();
                f.write(reinterpret_cast<const char*>(&ev.event_time), sizeof(ev.event_time));
                f.write(reinterpret_cast<const char*>(&ev.type), sizeof(ev.type));
                f.write(reinterpret_cast<const char*>(&ev.priority), sizeof(ev.priority));
                queue_copy.pop();
            }
        }

        void LoadState(std::ifstream & f) {

            Clear();

            size_t n_system_events = 0;
            f.read(reinterpret_cast<char*>(&n_system_events), sizeof(n_system_events));
            if(!f || n_system_events > 100000000)
                throw std::runtime_error("SystemEventScheduler::LoadState: invalid event count or truncated checkpoint.");

            for(size_t i = 0; i < n_system_events; ++i) {
                SystemEvent ev{};
                f.read(reinterpret_cast<char*>(&ev.event_time), sizeof(ev.event_time));
                f.read(reinterpret_cast<char*>(&ev.type), sizeof(ev.type));
                f.read(reinterpret_cast<char*>(&ev.priority), sizeof(ev.priority));
                if(!f)
                    throw std::runtime_error("SystemEventScheduler::LoadState: truncated event data.");
                events.push(ev);
            }
        }
};

#endif