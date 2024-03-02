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
            sycl::int4 *deviceNewClient;

            int *deviceInputData;
            int *deviceInserts;
            int *deviceInsertsClone;
            int *deviceInsertsIdx;

            int *deviceInputIdx;

            int *deviceTotalNewInserts;
            int *hostTotalNewInserts;

            int *hostInputData;
            int *hostInserts;

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

            void clear();
            int insert(int epoch);

            void set(organisation::data &mappings, inputs::input &source);
            std::vector<value> get();

        public:
            void copy(::organisation::schema **source, int source_size);

        protected:
            void outputarb(int *source, int length);

        protected:
            void makeNull();
            void cleanup();
        };
    };
};

#endif