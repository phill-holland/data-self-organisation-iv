#include "parallel/inserts.hpp"

sycl::int4 MapClientIdx(const int index, const sycl::int4 dimensions)
{
    int n = dimensions.x() * dimensions.y();
    int r = index % n;
    float z = (float)((index / n));

    int j = r % dimensions.x();
    float y = (float)((r / dimensions.x()));
    float x = (float)(j);

    return { x, y, z, index };
}

void organisation::parallel::inserts::reset(::parallel::device &dev, 
                                            ::parallel::queue *q, 
                                            parameters &settings)
{
    init = false; cleanup();

    this->dev = &dev;
    this->queue = q;
    this->settings = settings;
    this->length = settings.clients();

    sycl::queue &qt = ::parallel::queue(dev).get();

    deviceNewPositions = sycl::malloc_device<sycl::float4>(length, qt);
    if(deviceNewPositions == NULL) return;

    deviceNewValues = sycl::malloc_device<int>(length, qt);
    if(deviceNewValues == NULL) return;

    deviceNewMovementPatternIdx = sycl::malloc_device<int>(length, qt);
    if(deviceNewMovementPatternIdx == NULL) return;

    deviceNewClient = sycl::malloc_device<sycl::int4>(length, qt);
    if(deviceNewClient == NULL) return;

    deviceInputData = sycl::malloc_device<int>(settings.max_input_data * settings.epochs(), qt);
    if(deviceInputData == NULL) return;

    deviceInsertsDelay = sycl::malloc_device<int>(settings.max_inserts * settings.clients(), qt);
    if(deviceInsertsDelay == NULL) return;

    deviceInsertsDelayClone = sycl::malloc_device<int>(settings.max_inserts * settings.clients(), qt);
    if(deviceInsertsDelayClone == NULL) return;

    deviceInsertsStartingPosition = sycl::malloc_device<sycl::float4>(settings.max_inserts * settings.clients(), qt);
    if(deviceInsertsStartingPosition == NULL) return;

    deviceInsertsMovementPatternIdx = sycl::malloc_device<int>(settings.max_inserts * settings.clients(), qt);
    if(deviceInsertsMovementPatternIdx == NULL) return;

    deviceInputIdx = sycl::malloc_device<int>(settings.clients(), qt);
    if(deviceInputIdx == NULL) return;

    deviceTotalNewInserts = sycl::malloc_device<int>(1, qt);
    if(deviceTotalNewInserts == NULL) return;

    hostTotalNewInserts = sycl::malloc_host<int>(1, qt);
    if(hostTotalNewInserts == NULL) return;

    hostInputData = sycl::malloc_host<int>(settings.max_input_data * settings.input.size(), qt);
    if(hostInputData == NULL) return;

    hostInsertsDelay = sycl::malloc_host<int>(settings.max_inserts * settings.host_buffer, qt);
    if(hostInsertsDelay == NULL) return;

    hostInsertsStartingPosition = sycl::malloc_host<sycl::float4>(settings.max_inserts * settings.host_buffer, qt);
    if(hostInsertsStartingPosition == NULL) return;

    hostInsertsMovementPatternIdx = sycl::malloc_host<int>(settings.max_inserts * settings.host_buffer, qt);
    if(hostInsertsMovementPatternIdx == NULL) return;

    clear();

    init = true;
}

void organisation::parallel::inserts::clear()
{
    sycl::queue& qt = ::parallel::queue::get_queue(*dev, queue);

    std::vector<sycl::event> events;

    events.push_back(qt.memset(deviceNewPositions, 0, sizeof(sycl::float4) * length));
    events.push_back(qt.memset(deviceNewValues, -1, sizeof(int) * length));
    events.push_back(qt.memset(deviceNewClient, 0, sizeof(sycl::int4) * length));
    events.push_back(qt.memset(deviceInputIdx, 0, sizeof(int) * settings.clients()));

    sycl::event::wait(events);
}

int organisation::parallel::inserts::insert(int epoch)
{
    sycl::queue& qt = ::parallel::queue::get_queue(*dev, queue);
    sycl::range num_items{(size_t)length};

    qt.memset(deviceTotalNewInserts, 0, sizeof(int)).wait();

    auto epoch_offset = (epoch > settings.epochs() ? settings.epochs() : epoch) * settings.max_input_data;

    sycl::float4 starting = { (float)settings.starting.x, (float)settings.starting.y, (float)settings.starting.z, 0.0f };
    sycl::int4 dim_clients = { settings.dim_clients.x, settings.dim_clients.y, settings.dim_clients.z, 0.0f };

    qt.submit([&](auto &h) 
    {        
        auto _inserts = deviceInsertsDelay;
        auto _insertsClone = deviceInsertsDelayClone;

        auto _insertsStartingPosition = deviceInsertsStartingPosition;
        auto _insertsMovementPatternIdx = deviceInsertsMovementPatternIdx;

        auto _inputData = deviceInputData;
        auto _inputIdx = deviceInputIdx;

        auto _totalNewInserts = deviceTotalNewInserts;
        
        auto _values = deviceNewValues;
        auto _positions = deviceNewPositions;
        auto _movementPatternIdx = deviceNewMovementPatternIdx;
        auto _clients = deviceNewClient;

        //auto _starting = starting;

        auto _epoch_offset = epoch_offset;

        auto _max_inserts = settings.max_inserts;
        auto _dim_clients = dim_clients;

        auto _length = length;

        h.parallel_for(num_items, [=](auto client) 
        {
            int offset = (client * _max_inserts);

            for(int i = 0; i < _max_inserts; ++i)
            {
                if(_insertsMovementPatternIdx[i + offset] != -1)
                {
                    if(_inputData[_inputIdx[client] + epoch_offset] == -1) return;
                                        
                    //int a = _insertsIdx[client];            
                    //_inserts[a + offset]--;
                    _inserts[offset]--;

                    //if(_inserts[a + offset] < 0)
                    if(_inserts[offset] < 0)
                    {                
                        //_insertsIdx[client]++;
                        int b = _inputIdx[client];
                        int newValueToInsert = _inputData[b + epoch_offset];
                        _inputIdx[client]++;

                        //if(_inserts[_insertsIdx[client]] == -1)
                            //_insertsIdx[client] = 0;

                        //_inserts[a + offset] = _insertsClone[a + offset];
                        _inserts[offset] = _insertsClone[offset];

                        cl::sycl::atomic_ref<int, cl::sycl::memory_order::relaxed, 
                                                    sycl::memory_scope::device, 
                                                    sycl::access::address_space::ext_intel_global_device_space> ar(_totalNewInserts[0]);

                        int dest = ar.fetch_add(1);
                        if(dest < _length)                
                        {
                            _values[dest] = newValueToInsert;
                            _positions[dest] = _insertsStartingPosition[offset + i];//starting;
                            _movementPatternIdx[dest] = _insertsMovementPatternIdx[offset + i];
                            _clients[dest] = MapClientIdx(client, _dim_clients);
                        }
                    }
                }
            }
        });
    }).wait();

    qt.memcpy(hostTotalNewInserts, deviceTotalNewInserts, sizeof(int)).wait();
    return hostTotalNewInserts[0];
}

void organisation::parallel::inserts::set(organisation::data &mappings, inputs::input &source)
{
    memset(hostInputData, -1, sizeof(int) * settings.max_input_data * settings.epochs());

    int offset = 0;
    for(int i = 0; i < source.size(); ++i)
    {        
        inputs::epoch epoch;
        if(source.get(epoch, i))
        {
            std::vector<int> temp = mappings.get(epoch.input);
            int len = temp.size();
            if(len > settings.max_input_data) len = settings.max_input_data;

            for(int j = 0; j < len; ++j)
            {
                hostInputData[j + offset] = temp[j];
            }

            offset += settings.max_input_data;
        }
    }
    
    sycl::queue& qt = ::parallel::queue::get_queue(*dev, queue);
    qt.memcpy(deviceInputData, hostInputData, sizeof(int) * settings.max_input_data * settings.epochs()).wait();
}

std::vector<organisation::parallel::value> organisation::parallel::inserts::get()
{
    std::vector<value> result;

    int totals = hostTotalNewInserts[0];
    if(totals > 0)
    {
        std::vector<int> values = dev->get(deviceNewValues, totals);
        std::vector<sycl::int4> clients = dev->get(deviceNewClient, totals);
        std::vector<sycl::float4> positions = dev->get(deviceNewPositions, totals);

        if((values.size() == totals)&&(clients.size() == totals)&&(positions.size() == totals))
        {
            int len = values.size();
            for(int i = 0; i < len; ++i)
            {
                value temp;

                temp.data = values[i];
                temp.client = clients[i].w();
                temp.position = point(positions[i].x(), positions[i].y(), positions[i].z());

                result.push_back(temp);
            }
        }
    }

    return result;
}

void organisation::parallel::inserts::copy(::organisation::schema **source, int source_size)
{
    memset(hostInsertsDelay, -1, sizeof(int) * settings.max_inserts * settings.host_buffer);
    memset(hostInsertsStartingPosition, 0, sizeof(sycl::float4) * settings.max_inserts * settings.host_buffer);
    memset(hostInsertsMovementPatternIdx, -1, sizeof(int) * settings.max_inserts * settings.host_buffer);

    sycl::queue& qt = ::parallel::queue::get_queue(*dev, queue);
    sycl::range num_items{(size_t)settings.clients()};

    int dest_index = 0;
    int index = 0;

    for(int source_index = 0; source_index < source_size; ++source_index)
    {
        organisation::program *prog = &source[source_index]->prog;

        int i_count = 0;
        for(auto &it: prog->insert.values)
        {
            hostInsertsDelay[i_count + (index * settings.max_inserts)] = it.delay;
            hostInsertsStartingPosition[i_count + (index * settings.max_inserts)] = { (float)it.starting.x, (float)it.starting.y, (float)it.starting.z, 0.0f };
            hostInsertsMovementPatternIdx[i_count + (index * settings.max_inserts)] = it.movementPatternIdx;

            ++i_count;
            if(i_count >= settings.max_inserts) break;
        }

        ++index;
        if(index >= settings.host_buffer)
        {
            std::vector<sycl::event> events;

            events.push_back(qt.memcpy(&deviceInsertsDelay[dest_index * settings.max_inserts], hostInsertsDelay, sizeof(int) * settings.max_inserts * index));
            events.push_back(qt.memcpy(&deviceInsertsStartingPosition[dest_index * settings.max_inserts], hostInsertsStartingPosition, sizeof(sycl::float4) * settings.max_inserts * index));
            events.push_back(qt.memcpy(&deviceInsertsMovementPatternIdx[dest_index * settings.max_inserts], hostInsertsMovementPatternIdx, sizeof(int) * settings.max_inserts * index));

            memset(hostInsertsDelay, -1, sizeof(int) * settings.max_inserts * settings.host_buffer);
            memset(hostInsertsStartingPosition, 0, sizeof(sycl::float4) * settings.max_inserts * settings.host_buffer);
            memset(hostInsertsMovementPatternIdx, -1, sizeof(int) * settings.max_inserts * settings.host_buffer);

            sycl::event::wait(events);

            dest_index += settings.host_buffer;
            index = 0;            
        }
    }

    if(index > 0)
    {
        std::vector<sycl::event> events;

        events.push_back(qt.memcpy(&deviceInsertsDelay[dest_index * settings.max_inserts], hostInsertsDelay, sizeof(int) * settings.max_inserts * index));
        events.push_back(qt.memcpy(&deviceInsertsStartingPosition[dest_index * settings.max_inserts], hostInsertsStartingPosition, sizeof(sycl::float4) * settings.max_inserts * index));
        events.push_back(qt.memcpy(&deviceInsertsMovementPatternIdx[dest_index * settings.max_inserts], hostInsertsMovementPatternIdx, sizeof(int) * settings.max_inserts * index));

        sycl::event::wait(events);
    }   

    qt.memcpy(deviceInsertsDelayClone, deviceInsertsDelay, sizeof(int) * settings.max_inserts * settings.clients()).wait();
}

void organisation::parallel::inserts::outputarb(int *source, int length)
{
	int *temp = new int[length];
	if (temp == NULL) return;

    sycl::queue q = ::parallel::queue(*dev).get();

    q.memcpy(temp, source, sizeof(int) * length).wait();

    std::string result("");
	for (int i = 0; i < length; ++i)
	{
		if ((temp[i] != -1)&&(temp[i]!=0))
		{
			result += std::string("[");
			result += std::to_string(i);
			result += std::string("]");
			result += std::to_string(temp[i]);
			result += std::string(",");
		}
	}
	result += std::string("\r\n");
	
    std::cout << result;

	delete[] temp;
}

void organisation::parallel::inserts::makeNull()
{
    dev = NULL;

    deviceNewPositions = NULL;
    deviceNewValues = NULL;
    deviceNewMovementPatternIdx = NULL;
    deviceNewClient = NULL;
    
    deviceInputData = NULL;
    deviceInsertsDelay = NULL;
    deviceInsertsDelayClone = NULL;
    deviceInsertsStartingPosition = NULL;
    deviceInsertsMovementPatternIdx = NULL;
    
    deviceInputIdx = NULL;

    deviceTotalNewInserts = NULL;

    hostTotalNewInserts = NULL;    
    hostInputData = NULL;

    hostInsertsDelay = NULL;
    hostInsertsStartingPosition = NULL;
    hostInsertsMovementPatternIdx = NULL;
}

void organisation::parallel::inserts::cleanup()
{
    if(dev != NULL) 
    {   
        sycl::queue q = ::parallel::queue(*dev).get();

        if(hostInsertsMovementPatternIdx != NULL) sycl::free(hostInsertsMovementPatternIdx, q);
        if(hostInsertsStartingPosition != NULL) sycl::free(hostInsertsStartingPosition, q);
        if(hostInsertsDelay != NULL) sycl::free(hostInsertsDelay, q);
        if(hostInputData != NULL) sycl::free(hostInputData, q);
        if(hostTotalNewInserts != NULL) sycl::free(hostTotalNewInserts, q);        
        if(deviceTotalNewInserts != NULL) sycl::free(deviceTotalNewInserts, q);
        if(deviceInputIdx != NULL) sycl::free(deviceInputIdx, q);

        if(deviceInsertsMovementPatternIdx != NULL) sycl::free(deviceInsertsMovementPatternIdx, q);
        if(deviceInsertsStartingPosition != NULL) sycl::free(deviceInsertsStartingPosition, q);
        if(deviceInsertsDelayClone != NULL) sycl::free(deviceInsertsDelayClone, q);
        if(deviceInsertsDelay != NULL) sycl::free(deviceInsertsDelay, q);
        if(deviceInputData != NULL) sycl::free(deviceInputData, q);
        if(deviceNewClient != NULL) sycl::free(deviceNewClient, q);
        if(deviceNewMovementPatternIdx != NULL) sycl::free(deviceNewMovementPatternIdx, q);
        if(deviceNewValues != NULL) sycl::free(deviceNewValues, q);
        if(deviceNewPositions != NULL) sycl::free(deviceNewPositions, q);          
    }
}
