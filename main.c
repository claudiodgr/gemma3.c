/*
 * main.c - CLI interface for Gemma 3 inference
 *
 * Usage:
 *   ./gemma3 -m <model_dir> -p "Your prompt here"
 *   ./gemma3 -m <model_dir> -i -s "System prompt"
 */

#include "gemma3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>

/* ============================================================================
 * Configuration
 * ========================================================================== */

typedef struct {
    const char *model_dir;
    const char *prompt;
    const char *system_prompt;
    int interactive;
    int max_tokens;
    float temperature;
    int top_k;
    float top_p;
    int seed;
    int context_size;
    int verbose;
} cli_config;

static cli_config default_cli_config(void) {
    return (cli_config){
        .model_dir = NULL,
        .prompt = NULL,
        .system_prompt = "You are a helpful assistant.",
        .interactive = 0,
        .max_tokens = 512,
        .temperature = 0.7f,
        .top_k = 50,
        .top_p = 0.9f,
        .seed = -1,
        .context_size = 8192,
        .verbose = 0,
    };
}

/* ============================================================================
 * Signal Handling
 * ========================================================================== */

static volatile int g_interrupted = 0;

static void signal_handler(int sig) {
    (void)sig;
    g_interrupted = 1;
}

/* ============================================================================
 * Streaming Callback
 * ========================================================================== */

static int stream_callback(int token_id, const char *token_str, void *user_data) {
    (void)token_id;
    (void)user_data;

    if (g_interrupted) {
        return 1;  /* Stop generation */
    }

    /* Handle special tokens */
    if (token_str && token_str[0] != '\0') {
        /* Convert sentencepiece space marker to actual space */
        const char *ptr = token_str;
        while (*ptr) {
            /* Check for ▁ (0xE2 0x96 0x81) */
            if ((unsigned char)ptr[0] == 0xE2 &&
                (unsigned char)ptr[1] == 0x96 &&
                (unsigned char)ptr[2] == 0x81) {
                putchar(' ');
                ptr += 3;
            } else if (ptr[0] == '<' && ptr[1] == '0' && ptr[2] == 'x') {
                /* Byte token - skip for now */
                while (*ptr && *ptr != '>') ptr++;
                if (*ptr == '>') ptr++;
            } else {
                putchar(*ptr);
                ptr++;
            }
        }
        fflush(stdout);
    }

    return 0;
}

/* ============================================================================
 * Help and Usage
 * ========================================================================== */

static void print_usage(const char *prog) {
    fprintf(stderr, "Gemma 3 4B Inference - Pure C Implementation\n");
    fprintf(stderr, "Version: %s\n\n", gemma3_version());
    fprintf(stderr, "Usage: %s [options]\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m, --model <path>      Path to model directory (required)\n");
    fprintf(stderr, "  -p, --prompt <text>     Input prompt for generation\n");
    fprintf(stderr, "  -i, --interactive       Interactive chat mode\n");
    fprintf(stderr, "  -s, --system <text>     System prompt for chat mode\n");
    fprintf(stderr, "  -n, --max-tokens <n>    Maximum tokens to generate (default: 512)\n");
    fprintf(stderr, "  -t, --temperature <f>   Sampling temperature (default: 0.7)\n");
    fprintf(stderr, "  -k, --top-k <n>         Top-k sampling (default: 50, 0=disabled)\n");
    fprintf(stderr, "  --top-p <f>             Top-p sampling (default: 0.9)\n");
    fprintf(stderr, "  --seed <n>              Random seed (-1 for random)\n");
    fprintf(stderr, "  -c, --context <n>       Context size (default: 8192)\n");
    fprintf(stderr, "  -v, --verbose           Verbose output\n");
    fprintf(stderr, "  -h, --help              Show this help message\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  %s -m ./gemma-3-4b-it -p \"Hello, how are you?\"\n", prog);
    fprintf(stderr, "  %s -m ./gemma-3-4b-it -i\n", prog);
    fprintf(stderr, "  %s -m ./gemma-3-4b-it -i -s \"You are a pirate.\"\n", prog);
}

/* ============================================================================
 * Argument Parsing
 * ========================================================================== */

static int parse_args(int argc, char **argv, cli_config *config) {
    *config = default_cli_config();

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];

        if (strcmp(arg, "-m") == 0 || strcmp(arg, "--model") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: -m requires an argument\n");
                return 0;
            }
            config->model_dir = argv[i];
        } else if (strcmp(arg, "-p") == 0 || strcmp(arg, "--prompt") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: -p requires an argument\n");
                return 0;
            }
            config->prompt = argv[i];
        } else if (strcmp(arg, "-i") == 0 || strcmp(arg, "--interactive") == 0) {
            config->interactive = 1;
        } else if (strcmp(arg, "-s") == 0 || strcmp(arg, "--system") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: -s requires an argument\n");
                return 0;
            }
            config->system_prompt = argv[i];
        } else if (strcmp(arg, "-n") == 0 || strcmp(arg, "--max-tokens") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: -n requires an argument\n");
                return 0;
            }
            config->max_tokens = atoi(argv[i]);
        } else if (strcmp(arg, "-t") == 0 || strcmp(arg, "--temperature") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: -t requires an argument\n");
                return 0;
            }
            config->temperature = atof(argv[i]);
        } else if (strcmp(arg, "-k") == 0 || strcmp(arg, "--top-k") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: -k requires an argument\n");
                return 0;
            }
            config->top_k = atoi(argv[i]);
        } else if (strcmp(arg, "--top-p") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: --top-p requires an argument\n");
                return 0;
            }
            config->top_p = atof(argv[i]);
        } else if (strcmp(arg, "--seed") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: --seed requires an argument\n");
                return 0;
            }
            config->seed = atoi(argv[i]);
        } else if (strcmp(arg, "-c") == 0 || strcmp(arg, "--context") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: -c requires an argument\n");
                return 0;
            }
            config->context_size = atoi(argv[i]);
        } else if (strcmp(arg, "-v") == 0 || strcmp(arg, "--verbose") == 0) {
            config->verbose = 1;
        } else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "Error: Unknown option '%s'\n", arg);
            return 0;
        }
    }

    if (!config->model_dir) {
        fprintf(stderr, "Error: Model directory (-m) is required\n");
        return 0;
    }

    if (!config->interactive && !config->prompt) {
        fprintf(stderr, "Error: Either -p (prompt) or -i (interactive) is required\n");
        return 0;
    }

    return 1;
}

/* ============================================================================
 * Single Prompt Mode
 * ========================================================================== */

static int run_single_prompt(gemma3_ctx *ctx, const cli_config *config) {
    gemma3_gen_params params = {
        .max_tokens = config->max_tokens,
        .temperature = config->temperature,
        .top_k = config->top_k,
        .top_p = config->top_p,
        .seed = config->seed,
        .stop_on_eos = 1,
    };

    g_interrupted = 0;

    char *response = gemma3_generate(ctx, config->prompt, &params,
                                     stream_callback, NULL);
    printf("\n");

    if (!response) {
        fprintf(stderr, "Error: Generation failed: %s\n", gemma3_get_error());
        return 1;
    }

    free(response);
    return 0;
}

/* ============================================================================
 * Interactive Chat Mode
 * ========================================================================== */

#define MAX_INPUT_LEN 4096
#define MAX_MESSAGES 100

static int run_interactive(gemma3_ctx *ctx, const cli_config *config) {
    printf("Gemma 3 Interactive Chat\n");
    printf("Type 'quit' or 'exit' to end, 'clear' to reset conversation\n");
    printf("System: %s\n", config->system_prompt);
    printf("---\n\n");

    gemma3_message messages[MAX_MESSAGES];
    int num_messages = 0;

    /* Add system message */
    if (config->system_prompt && strlen(config->system_prompt) > 0) {
        messages[num_messages].role = GEMMA3_ROLE_SYSTEM;
        messages[num_messages].content = config->system_prompt;
        num_messages++;
    }

    char input[MAX_INPUT_LEN];

    while (1) {
        /* Prompt */
        printf("You: ");
        fflush(stdout);

        /* Read input */
        if (!fgets(input, sizeof(input), stdin)) {
            printf("\n");
            break;
        }

        /* Remove trailing newline */
        int len = strlen(input);
        if (len > 0 && input[len - 1] == '\n') {
            input[len - 1] = '\0';
            len--;
        }

        /* Skip empty input */
        if (len == 0) continue;

        /* Check for commands */
        if (strcmp(input, "quit") == 0 || strcmp(input, "exit") == 0) {
            printf("Goodbye!\n");
            break;
        }

        if (strcmp(input, "clear") == 0) {
            num_messages = 0;
            if (config->system_prompt && strlen(config->system_prompt) > 0) {
                messages[num_messages].role = GEMMA3_ROLE_SYSTEM;
                messages[num_messages].content = config->system_prompt;
                num_messages++;
            }
            gemma3_reset_cache(ctx);
            printf("[Conversation cleared]\n\n");
            continue;
        }

        /* Check message limit */
        if (num_messages >= MAX_MESSAGES - 1) {
            printf("[Warning: Maximum messages reached, clearing history]\n");
            num_messages = 0;
            if (config->system_prompt && strlen(config->system_prompt) > 0) {
                messages[num_messages].role = GEMMA3_ROLE_SYSTEM;
                messages[num_messages].content = config->system_prompt;
                num_messages++;
            }
            gemma3_reset_cache(ctx);
        }

        /* Add user message */
        char *user_input = strdup(input);
        if (!user_input) {
            fprintf(stderr, "Error: Out of memory\n");
            break;
        }
        messages[num_messages].role = GEMMA3_ROLE_USER;
        messages[num_messages].content = user_input;
        num_messages++;

        /* Generate response */
        gemma3_gen_params params = {
            .max_tokens = config->max_tokens,
            .temperature = config->temperature,
            .top_k = config->top_k,
            .top_p = config->top_p,
            .seed = config->seed,
            .stop_on_eos = 1,
        };

        g_interrupted = 0;
        printf("\nGemma: ");
        fflush(stdout);

        char *response = gemma3_chat(ctx, messages, num_messages, &params,
                                     stream_callback, NULL);
        printf("\n\n");

        if (!response) {
            fprintf(stderr, "[Error: Generation failed: %s]\n\n", gemma3_get_error());
            /* Remove the failed user message */
            free((char *)messages[num_messages - 1].content);
            num_messages--;
            continue;
        }

        /* Add assistant response to history */
        messages[num_messages].role = GEMMA3_ROLE_MODEL;
        messages[num_messages].content = response;
        num_messages++;
    }

    /* Cleanup message history */
    for (int i = 0; i < num_messages; i++) {
        if (messages[i].role != GEMMA3_ROLE_SYSTEM) {
            free((char *)messages[i].content);
        }
    }

    return 0;
}

/* ============================================================================
 * Main
 * ========================================================================== */

int main(int argc, char **argv) {
    cli_config config;

    if (!parse_args(argc, argv, &config)) {
        print_usage(argv[0]);
        return 1;
    }

    /* Set up signal handler */
    signal(SIGINT, signal_handler);

    /* Load model */
    gemma3_ctx *ctx = gemma3_load_dir_ex(config.model_dir, config.context_size);
    if (!ctx) {
        fprintf(stderr, "Error: Failed to load model: %s\n", gemma3_get_error());
        return 1;
    }

    if (config.verbose) {
        const gemma3_config *model_config = gemma3_get_config(ctx);
        printf("Model configuration:\n");
        printf("  Vocab size: %d\n", model_config->vocab_size);
        printf("  Hidden size: %d\n", model_config->hidden_size);
        printf("  Layers: %d\n", model_config->num_layers);
        printf("  Heads: %d (KV: %d)\n", model_config->num_heads, model_config->num_kv_heads);
        printf("  Head dim: %d\n", model_config->head_dim);
        printf("  Context: %d\n", model_config->max_context);
        printf("\n");
    }

    int result;
    if (config.interactive) {
        result = run_interactive(ctx, &config);
    } else {
        result = run_single_prompt(ctx, &config);
    }

    gemma3_free(ctx);
    return result;
}
