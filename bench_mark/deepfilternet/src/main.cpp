#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "deep_filter.h"
#include "openvino/openvino.hpp"
#include "deepfilter.hpp"

#define DEEPFILTERNET_PROCESSFRAMELENGTH 480
#define DEEPFILTERNET_ONNX "/Users/andy.chen/Documents/Opensource/DeepFilterNet/bench_mark/deepfilternet/model/onnx/DeepFilterNet3_onnx.tar.gz"
#define DEEPFILTERNET_OPENVINO "/Users/andy.chen/Documents/Opensource/DeepFilterNet/bench_mark/deepfilternet/model/openvino/"

// Read the WAV file header
typedef struct header_
{
    char riff[4];
    uint32_t file_size;
    char wave[4];
    char fmt[4];
    uint32_t fmt_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char subchunk2ID[4];
    unsigned int subchunk2Size;
} header_wav;

void read_wav_info(const char *wav_path, int &bytesize)
{
    // Open the WAV file
    FILE *wav_file = fopen(wav_path, "rb");
    if (wav_file == NULL)
    {
        printf("Failed to open WAV file\n");
        return;
    }

    header_wav header;

    fread(&header, sizeof(header), 1, wav_file);
    bytesize = header.subchunk2Size;

    // Print the WAV file info
    printf("WAV File Info:\n");
    printf("  File Size: %u bytes\n", header.file_size);
    printf("  Audio Format: %u\n", header.audio_format);
    printf("  Number of Channels: %u\n", header.num_channels);
    printf("  Sample Rate: %u Hz\n", header.sample_rate);
    printf("  Byte Rate: %u bytes/sec\n", header.byte_rate);
    printf("  Block Align: %u bytes\n", header.block_align);
    printf("  Bits per Sample: %u bits\n", header.bits_per_sample);
    printf(" Subchunk2 Size: %u bytes\n", header.subchunk2Size);
    printf(" bytesize: %d\n", bytesize);

    // Close the WAV file
    fclose(wav_file);
}

void *Deepfilter_create(const char *path, float level)
{
    return static_cast<void *>(df_create(path, level, nullptr));
}

void Deepfilter_free(void *df)
{
    df_free(static_cast<DFState *>(df));
}

float ProcessFrameAudioDeepFilterWrapper(void *df, int channel_num, void *input)
{
    printf("onnx process the audio frame\n");
    int16_t *midbuffer = (int16_t *)input;
    float leftbuffer[DEEPFILTERNET_PROCESSFRAMELENGTH];
    for (int i = 0; i < DEEPFILTERNET_PROCESSFRAMELENGTH; i++)
    {
        leftbuffer[i] = midbuffer[channel_num * i] / 32767.0f;
    }
    float l_snr = df_process_frame(static_cast<DFState *>(df), leftbuffer, leftbuffer);
    for (int i = 0; i < DEEPFILTERNET_PROCESSFRAMELENGTH; i++)
    {
        midbuffer[channel_num * i] = (int16_t)(leftbuffer[i] * 32767.0);
        if (channel_num == 2)
        {
            midbuffer[channel_num * i + 1] = midbuffer[channel_num * i];
        }
    }
    return 0.0f;
}

bool ProcessFrameAudioOpenVinoWrapper(DeepFilter *_df, int channel_num, void *input, float atten_lim, bool _bDF3_post_filter)
{
    printf("openvino process the audio frame\n");
    int16_t *midbuffer = (int16_t *)input;
    float leftbuffer[DEEPFILTERNET_PROCESSFRAMELENGTH];
    for (int i = 0; i < DEEPFILTERNET_PROCESSFRAMELENGTH; i++)
    {
        leftbuffer[i] = midbuffer[channel_num * i] / 32767.0f;
    }
    torch::Tensor input_wav_tensor = torch::from_blob(leftbuffer, {1, (int64_t)DEEPFILTERNET_PROCESSFRAMELENGTH});

    auto wav = _df->filter(input_wav_tensor, atten_lim, 20, _bDF3_post_filter, nullptr, nullptr);
    if (!wav)
    {
        std::cout << "!wav -- returning false" << std::endl;
        return false;
    }

    std::vector<float> &vec = *wav;
    for (int i = 0; i < DEEPFILTERNET_PROCESSFRAMELENGTH; i++)
    {
        midbuffer[channel_num * i] = (int16_t)(vec[i] * 32767.0);
        if (channel_num == 2)
        {
            midbuffer[channel_num * i + 1] = midbuffer[channel_num * i];
        }
    }
    return true;
}

bool ProcessFileAudioOpenVinoWrapper(DeepFilter *_df, int channel_num, void *input, int samplepoints, float atten_lim, bool _bDF3_post_filter)
{
    printf("openvino process the audio frame\n");
    int16_t *midbuffer = (int16_t *)input;
    float leftbuffer[samplepoints/2];
    for (int i = 0; i < samplepoints/2; i++)
    {
        leftbuffer[i] = midbuffer[channel_num * i] / 32767.0f;
    }
    torch::Tensor input_wav_tensor = torch::from_blob(leftbuffer, {1, (int64_t)(samplepoints/2)});

    auto wav = _df->filter(input_wav_tensor, atten_lim, 20, _bDF3_post_filter, nullptr, nullptr);
    if (!wav)
    {
        std::cout << "!wav -- returning false" << std::endl;
        return false;
    }

    std::vector<float> &vec = *wav;
    for (int i = 0; i < samplepoints/2; i++)
    {
        midbuffer[channel_num * i] = (int16_t)(vec[i] * 32767.0);
        if (channel_num == 2)
        {
            midbuffer[channel_num * i + 1] = midbuffer[channel_num * i];
        }
    }
    return true;
}

int main(int argc, char *argv[])
{
    if (argc < 7)
    {
        printf("Usage: %s -i <input_file> -m <model> -o <output_file>\n", argv[0]);
        return 1;
    }

    char *input_file;
    char *model_type;
    char *output_file;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-i") == 0)
        {
            input_file = argv[i + 1];
        }
        else if (strcmp(argv[i], "-m") == 0)
        {
            model_type = argv[i + 1];
        }
        else if (strcmp(argv[i], "-o") == 0)
        {
            output_file = argv[i + 1];
        }
    }

    // Read the WAV file info
    int bytesize = 0;
    read_wav_info(input_file, bytesize);

    // Open the WAV file
    FILE *input = fopen(input_file, "rb");
    if (input == NULL)
    {
        printf("Failed to open WAV file\n");
        return 1;
    }

    FILE *output = fopen(output_file, "wb");
    if (output == NULL)
    {
        printf("Failed to open output file\n");
        return 1;
    }
    header_wav header;

    fread(&header, sizeof(header), 1, input);
    // fwrite(&header, sizeof(header), 1, output);

    if (strcmp(model_type, "onnx") == 0)
    {
        printf("Using DeepFilterNet onnx model\n");
        void *df = Deepfilter_create(DEEPFILTERNET_ONNX, 100.0f);
        if (df == NULL)
        {
            printf("Failed to create DeepFilterNet model\n");
            return 1;
        }

        // Read the WAV file data
        int16_t midbuffer[DEEPFILTERNET_PROCESSFRAMELENGTH * 2];
        while (fread(midbuffer, sizeof(midbuffer), 1, input)>0)
        {
            ProcessFrameAudioDeepFilterWrapper(df, 2, midbuffer);
            fwrite(midbuffer, sizeof(midbuffer), 1, output);
            break;
            memset(midbuffer, 0, sizeof(midbuffer));
        }

        // Close the WAV file
        fclose(input);
        fclose(output);

        Deepfilter_free(df);
    }
    else if (strcmp(model_type, "openvino") == 0)
    {
        printf("Using DeepFilterNet openvino model\n");
        DeepFilter _df(DEEPFILTERNET_OPENVINO, "CPU");
        // Read the WAV file data
        int16_t midbuffer[DEEPFILTERNET_PROCESSFRAMELENGTH * 2];
        while (fread(midbuffer, sizeof(midbuffer), 1, input)>0)
        {
            ProcessFrameAudioOpenVinoWrapper(&_df, 2, midbuffer, 100.0f, false);
            fwrite(midbuffer, sizeof(midbuffer), 1, output);
            memset(midbuffer, 0, sizeof(midbuffer));
        }
        fclose(input);
        fclose(output);
    }
    else{
        printf("Invalid model type\n");
        return 1;
    }
    fclose(output);
    return 0;
}