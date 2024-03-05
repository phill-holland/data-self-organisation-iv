#include "genetic/templates/genetic.h"
#include "genetic/templates/serialiser.h"
#include "point.h"
#include "vector.h"
#include "data.h"
#include "parameters.h"
#include <vector>
#include <tuple>
#include <random>

#ifndef _ORGANISATION_GENETIC_MOVEMENT
#define _ORGANISATION_GENETIC_MOVEMENT

namespace organisation
{
    namespace genetic
    {
        namespace movements
        {
            class movement : public templates::genetic, public templates::serialiser
            {
                static std::mt19937_64 generator;

                int _min_movements, _max_movements;
                int _min_movement_patterns, _max_movement_patterns;

            public:
                std::vector<std::tuple<int,organisation::vector>> directions;

            public:
                movement(parameters &settings)
                {
                    _min_movements = settings.min_movements;
                    _max_movements = settings.max_movements;
                    _min_movement_patterns = settings.min_movement_patterns;
                    _max_movement_patterns = settings.max_movement_patterns;
                }

            public:
                size_t size() { return directions.size(); }

                void clear() 
                {
                    directions.clear();
                }

                bool empty() 
                {
                    return directions.empty();
                }

                int next(int index)
                {
                    if(index + 1 < directions.size()) return index + 1;
                    return 0;
                }

                void set(int pattern, std::vector<organisation::vector> &source);

                std::string serialise();
                void deserialise(std::string source);

                bool validate(data &source);
                
            public:
                void generate(data &source);
                bool mutate(data &source);
                void append(genetic *source, int src_start, int src_end);

            public:
                void copy(const movement &source);
                bool equals(const movement &source);
            };
        };
    };
};

#endif