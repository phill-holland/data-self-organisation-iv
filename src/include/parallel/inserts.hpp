#include <CL/sycl.hpp>
#include "parallel/device.hpp"
#include "parallel/queue.hpp"
#include "parallel/value.hpp"
#include "parameters.h"
#include "schema.h"

#ifndef _PARALLEL_INSERTS
#define _PARALLEL_INSERTS

namespace organisation
{    
    namespace parallel
    {        
        class program;

        class inserts
        {            
            friend class program;

            ::parallel::device *dev;
            ::parallel::queue *queue;

            sycl::float4 *deviceNewPositions;
            int *deviceNewValues;
            int *deviceNewMovementPatternIdx;
            sycl::int4 *deviceNewClient;

            int *deviceInputData;
            int *deviceInsertsDelay;
            int *deviceInsertsDelayClone;
            sycl::float4 *deviceInsertsStartingPosition;
            int *deviceInsertsMovementPatternIdx;

            sycl::float4 *deviceMovements;
            int *deviceMovementsCounts;

            int *deviceInputIdx;

            int *deviceTotalNewInserts;
            int *hostTotalNewInserts;

            int *hostInputData;

            int *hostInsertsDelay;
            sycl::float4 *hostInsertsStartingPosition;
            int *hostInsertsMovementPatternIdx;

            sycl::float4 *hostMovements;
            int *hostMovementsCounts;

            parameters settings;

            int length;

            bool init;

        public:
            inserts(::parallel::device &dev, 
                    ::parallel::queue *q,
                    parameters &settings) 
            { 
                makeNull(); 
                reset(dev, q, settings); 
            }
            ~inserts() { cleanup(); }

            bool initalised() { return init; }
            void reset(::parallel::device &dev, 
                       ::parallel::queue *q,
                       parameters &settings);

            void restart();
            void clear();
            
            int insert(int epoch);

            void set(organisation::data &mappings, inputs::input &source);
            std::vector<value> get();

        public:
            void copy(::organisation::schema **source, int source_size);

        protected:
            void outputarb(int *source, int length);
            void outputarb(sycl::float4 *source, int length);

        protected:
            void makeNull();
            void cleanup();
        };
    };
};

#endif