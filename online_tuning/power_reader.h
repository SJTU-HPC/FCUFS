#ifndef POWER_READER
#define POWER_READER
#include <stdio.h>
#include <stdlib.h>

// read these files to obtain power consumption
// /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
// /sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj
// /sys/class/powercap/intel-rapl/intel-rapl:1/energy_uj
// /sys/class/powercap/intel-rapl/intel-rapl:1/intel-rapl:1:0/energy_uj

#define MAX_FILE_PATH_LEN 1024
#define MAX_CASE_NUM 128

int _cpu_num;
long long int *_pkg_power;
long long int *_mem_power;
long long int _max_pkg_power;   // uj
long long int _max_mem_power;
char **_pkg_power_path;
char **_mem_power_path;
int _power_cur_case = 0;

long long int _read_file(const char *file_path) {
    long long int value;
    FILE *file = fopen(file_path, "r");
    if (file == NULL) {
        printf("Failed to open %s.\n", file_path);
        exit(-1);
    }

    fscanf(file, "%lld", &value);

    fclose(file);

    return value;
}

void power_reader_init(const int cpu_num) {
    _cpu_num = cpu_num;
    _pkg_power = (long long int*)malloc(sizeof(long long int*) * _cpu_num);
    _mem_power = (long long int*)malloc(sizeof(long long int*) * _cpu_num);
    _pkg_power_path = (char**)malloc(sizeof(char*) * _cpu_num);
    _mem_power_path = (char**)malloc(sizeof(char*) * _cpu_num);
    for (int i = 0; i < _cpu_num; ++i) {
        _pkg_power_path[i] = (char*)malloc(sizeof(char) * MAX_FILE_PATH_LEN);
        _mem_power_path[i] = (char*)malloc(sizeof(char) * MAX_FILE_PATH_LEN);
        sprintf(_pkg_power_path[i], "/sys/class/powercap/intel-rapl/intel-rapl:%d/energy_uj", i);
        sprintf(_mem_power_path[i], "/sys/class/powercap/intel-rapl/intel-rapl:%d/intel-rapl:%d:0/energy_uj", i, i);
    }
    _max_pkg_power = _read_file("/sys/class/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj");
    _max_mem_power = _read_file("/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/max_energy_range_uj");
}

void power_reader_start() {
    for (int i = 0; i < _cpu_num; ++i) {
        _pkg_power[i] = _read_file(_pkg_power_path[i]);
        _mem_power[i] = _read_file(_mem_power_path[i]);
    }
}

long long int _calc_power(
    const long long int start_value, 
    const long long int end_value, 
    const long long int max_value) 
{
    if (end_value >= start_value) {
        return end_value - start_value;
    } else {
        return max_value - start_value + end_value;
    }
}

long long int power_reader_end() {
    long long int total_power = 0;
    for (int i = 0; i < _cpu_num; ++i) {
        long long int pkg_power_e = _read_file(_pkg_power_path[i]);
        long long int mem_power_e = _read_file(_mem_power_path[i]);
        _pkg_power[i] = _calc_power(_pkg_power[i], pkg_power_e, _max_pkg_power);
        _mem_power[i] = _calc_power(_mem_power[i], mem_power_e, _max_mem_power);
        total_power += _pkg_power[i];
        total_power += _mem_power[i];
    }

    return total_power;
}

void power_reader_dump(const char *file_path) {
    FILE *fp = NULL;
    fp = fopen(file_path, "w");

    // save
    fprintf(fp, "%s,%s\n", "pkg_power", "mem_power");
    long long int t_pkg_power = 0;
    long long int t_mem_power = 0;
    for (int i = 0; i < _cpu_num; ++i) {
        t_pkg_power += _pkg_power[i];
        t_mem_power += _mem_power[i];
    }
    fprintf(fp, "%lld,%lld\n", t_pkg_power, t_mem_power);
    fclose(fp);
}

void power_reader_finalize() {
    // free
    for (int i = 0; i < _cpu_num; ++i) {
        free(_pkg_power_path[i]);
        free(_mem_power_path[i]);
    }
    free(_pkg_power);
    free(_mem_power);
    free(_pkg_power_path);
    free(_mem_power_path);
}

#endif
