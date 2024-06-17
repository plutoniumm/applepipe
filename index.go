package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"runtime"
	"strings"

	llama "github.com/go-skynet/go-llama.cpp"
	"github.com/gorilla/websocket"
)

var (
	threads   = 4
	tokens    = 128
	gpulayers = 0
	seed      = -1
)

func main() {
	var model string

	flags := flag.NewFlagSet(os.Args[0], flag.ExitOnError)
	flags.StringVar(&model, "m", "./models/7B/ggml-model-q4_0.bin", "path to q4_0.bin model file to load")
	flags.IntVar(&gpulayers, "ngl", 0, "Number of GPU layers to use")
	flags.IntVar(&threads, "t", runtime.NumCPU(), "number of threads to use during computation")
	flags.IntVar(&tokens, "n", 512, "number of tokens to predict")
	flags.IntVar(&seed, "s", -1, "predict RNG seed, -1 for random seed")

	err := flags.Parse(os.Args[1:])
	if err != nil {
		fmt.Printf("Parsing program arguments failed: %s", err)
		os.Exit(1)
	}

	l, err := llama.New(model, llama.EnableF16Memory, llama.SetContext(128), llama.EnableEmbeddings, llama.SetGPULayers(gpulayers))
	if err != nil {
		fmt.Println("Loading the model failed:", err.Error())
		os.Exit(1)
	}
	fmt.Printf("Model loaded successfully.\n")

	upgrader := websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool { return true },
	}

	http.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			fmt.Println("Failed to set websocket upgrade: ", err)
			return
		}
		handleWebSocket(conn, l)
	})

	http.ListenAndServe(":8080", nil)
}

func handleWebSocket(conn *websocket.Conn, l *llama.Llama) {
	defer conn.Close()
	for {
		_, message, err := conn.ReadMessage()
		if err != nil {
			fmt.Println("read:", err)
			break
		}
		text := string(message)
		fmt.Printf("Received: %s\n", text)

		var response strings.Builder
		_, err = l.Predict(text, llama.Debug, llama.SetTokenCallback(func(token string) bool {
			response.WriteString(token)
			return true
		}), llama.SetTokens(tokens), llama.SetThreads(threads), llama.SetTopK(90), llama.SetTopP(0.86), llama.SetStopWords("llama"), llama.SetSeed(seed))
		if err != nil {
			fmt.Println("Predict error:", err)
			conn.WriteMessage(websocket.TextMessage, []byte("Error: "+err.Error()))
			continue
		}
		embeds, err := l.Embeddings(text)
		if err != nil {
			fmt.Printf("Embeddings error: %s\n", err.Error())
			conn.WriteMessage(websocket.TextMessage, []byte("Error: "+err.Error()))
			continue
		}
		response.WriteString(fmt.Sprintf("\nEmbeddings: %v\n", embeds))

		err = conn.WriteMessage(websocket.TextMessage, []byte(response.String()))
		if err != nil {
			fmt.Println("write:", err)
			break
		}
	}
}

// readMultiLineInput reads input until an empty line is entered.
func readMultiLineInput(reader *bufio.Reader) string {
	var lines []string
	fmt.Print(">>> ")

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				os.Exit(0)
			}
			fmt.Printf("Reading the prompt failed: %s", err)
			os.Exit(1)
		}

		if len(strings.TrimSpace(line)) == 0 {
			break
		}

		lines = append(lines, line)
	}

	text := strings.Join(lines, "")
	fmt.Println("Sending", text)
	return text
}
