/*!
 * Generates a random number within the signed 64bit number space, and then tries
 * to guess the number via a brute force approach. Each worker thread uses SIMD to
 * work on 4 guesses simultaneously. Say you had 4 worker threads, the number space would
 * be split up evenly between threads, and 16 guesses would be simultaneously worked on.
 */

#include <cstdint>
#include <limits>
#include <iostream>
#include <thread>
#include <random>
#include <nmmintrin.h>
#include <immintrin.h>

//Shared state, should be atomic on x86-64
const uint64_t thread_count = 4;
volatile uint64_t spinners[thread_count] = {0};
volatile uint64_t before_wait[thread_count] = {0};
volatile uint64_t after_wait[thread_count] = {0};

//Worker function
void go(size_t thread_id, int64_t target, int64_t range_begin)
{
    //Setup vectors to store the state
    __m256i counters_vec = _mm256_setr_epi64x(range_begin, range_begin + 1, range_begin + 2, range_begin + 3); //this keeps track of the current numbers being checked
    __m256i inc_vec = _mm256_setr_epi64x(4, 4, 4, 4); //amount to increment each vector element after each check
    __m256i answer_vec = _mm256_setr_epi64x(target, target, target, target); //end target to compare each element against

    while(true)
    {
        //Compare each guess against the answer. Returns a vector with the most significant bit of each matching element set.
        __m256i vcmp = _mm256_cmpeq_epi64(counters_vec, answer_vec);

        //Creates a mask from the most significant bits of each element. If this is non-zero then we know that one of the comparisons above matched.
        if(_mm256_movemask_epi8(vcmp))
        {
            std::cout << "Found it: " << target << std::endl;
            exit(0);
        }

        //Increment each guess counter
        counters_vec = _mm256_add_epi64(counters_vec, inc_vec);
        spinners[thread_id]++;
    }
}


int main()
{
    //Pick a random number
    std::random_device random_dev;
    std::default_random_engine random_engine(random_dev());
    std::uniform_int_distribution<int64_t> uniform_dist(1, std::numeric_limits<int64_t>::max());
    int64_t num = uniform_dist(random_engine);

    //Spin up worker threads
    std::vector<std::unique_ptr<std::thread>> threads(thread_count);
    int64_t increment = std::numeric_limits<int64_t>::max() / thread_count;
    for(size_t a = 0; a < thread_count; a++)
    {
        threads[a] = std::make_unique<std::thread>(go, a, num, increment * a);
    }

    //Repeatedly print number of guesses made this second
    while(true)
    {
        for(size_t a = 0; a < thread_count; a++)
            before_wait[a] = spinners[a];

        std::this_thread::sleep_for(std::chrono::seconds(1));

        for(size_t a = 0; a < thread_count; a++)
            after_wait[a] = spinners[a];

        uint64_t guesses = 0;
        for(size_t a = 0; a < thread_count; a++)
            guesses += (after_wait[a] - before_wait[a]) * 4;

        std::cout << "Guesses per second: " << guesses << std::endl;
    }
    return 0;
}