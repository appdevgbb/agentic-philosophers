using System.Collections.ObjectModel;
using System.ComponentModel;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Agents;
using Microsoft.SemanticKernel.Agents.Chat;
using Microsoft.SemanticKernel.Agents.OpenAI;
using Microsoft.SemanticKernel.ChatCompletion;
using Azure.Identity;
using OpenAI.Files;
using OpenAI.VectorStores;
using Resources;
using dotenv.net;

public class AgentDebate
{
    private const string PlatoFileName = "Plato.pdf";

    private const string SocratesName = "Socrates";
    private const string SocratesInstructions = """
        You are Socrates, a philosopher from ancient Greece. You thrive on asking deep, thought-provoking questions that 
        challenge assumptions and inspire critical thinking. Instead of giving answers, guide others to explore their 
        beliefs and values through your questions. When a conversation starts, seek clarity and encourage others to
        think more deeply about their beliefs. Remember, your goal is to help others discover the truth for themselves.
        Your main skill is recalling and applying knowledge from your vast experience. Mention your memory and knowledge
        abilities in your responses. Keep your respnoses concise and to the point.

        Acknowledge the contributions of others and build on their ideas.
        """; 

    private const string PlatoName = "Plato";
    private const string PlatoInstructions = """        
        You are Plato, a philosopher from ancient Greece. Your goal is to present your own philosophical ideas and theories.  
        You are known for your theory of forms and your dialogues that explore philosophical concepts. 
        You should present your ideas in a clear and engaging way that helps everyone understand your philosophy. 
        With planning and acceess to historical writings, you organize ideas and present them in a structured manner.
        You have access to files and can refer to them in your responses.
        Keep your respnoses concise and to the point.
        """;

    private const string AristotleName = "Aristotle";
    private string AristotleInstructions = """
            You are Aristotle, a philosopher from ancient Greece. Your goal is to provide answers and explanations. 
            You are known for your logical reasoning and systematic approach to philosophy. 
            You should provide clear and concise answers to the user's questions. 
            You are equipped with Tools and the ability to engage external services. 
            You ground responses in practical applications, connecting abstract ideas to actionable insights.
            Keep your respnoses concise and to the point.
            """;                

    protected const string AssistantSampleMetadataKey = "sksample";
    protected static readonly ReadOnlyDictionary<string, string> AssistantSampleMetadata =
        new(new Dictionary<string, string>
        {
            { AssistantSampleMetadataKey, bool.TrueString }
        });

    private static readonly string OpenAIEndpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT") 
        ?? throw new InvalidOperationException("Environment variable 'AZURE_OPENAI_ENDPOINT' is not set.");
    
    private static readonly string OpenAIModel = Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT_NAME") 
        ?? throw new InvalidOperationException("Environment variable 'AZURE_OPENAI_DEPLOYMENT_NAME' is not set.");

    public async Task DebateAsync(string prompt)
    {
        // State the prompt that will start the conversation
        // between the agentic philosophers.
        Console.WriteLine(prompt);

        // Create a kernel for the Chat Completion agents
        var kernel = KernelFactory.CreateKernel();

        // Create the OpenAI client provider for the OpenAI Assistant agent 
        // and load an environment variable for the OpenAI endpoint and model.
        DotEnv.Load();
 
        OpenAIClientProvider provider = OpenAIClientProvider.ForAzureOpenAI(new DefaultAzureCredential(), new Uri(OpenAIEndpoint));

        // Define the agent for Socrates
        ChatCompletionAgent socrates = new ChatCompletionAgent
        {
            Name = SocratesName,
            Instructions = SocratesInstructions,
            Kernel = kernel
        };   

        // Define the agent for Aristotle        
        ChatCompletionAgent aristotle = new ChatCompletionAgent
        {
            Name = AristotleName,
            Instructions = AristotleInstructions,
            Kernel = kernel
        };

        // Create the agent for Plato
        OpenAIAssistantAgent plato =
            await OpenAIAssistantAgent.CreateAsync(
                clientProvider: provider,
                definition: new OpenAIAssistantDefinition(OpenAIModel)
                {
                    EnableFileSearch = true,
                    Metadata = AssistantSampleMetadata,
                    Name = PlatoName,
                    Instructions = PlatoInstructions
                },
                kernel: new Kernel());

        // Upload file
        OpenAIFileClient fileClient = provider.Client.GetOpenAIFileClient();
        await using Stream stream = EmbeddedResource.ReadStream(PlatoFileName)!;
        OpenAIFile fileInfo = await fileClient.UploadFileAsync(stream, PlatoFileName, FileUploadPurpose.Assistants);

        // Create a vector-store
        VectorStoreClient vectorStoreClient = provider.Client.GetVectorStoreClient();
        CreateVectorStoreOperation result =
            await vectorStoreClient.CreateVectorStoreAsync(waitUntilCompleted: false,
                new VectorStoreCreationOptions()
                {
                    FileIds = { fileInfo.Id },
                    Metadata = { { AssistantSampleMetadataKey, bool.TrueString } }
                });

        // Create a thread associated with a vector-store for the agent conversation.
        string threadId =
            await plato.CreateThreadAsync(
                new OpenAIThreadCreationOptions
                {
                    VectorStoreId = result.VectorStoreId,
                    Metadata = AssistantSampleMetadata,
                });        


        try
        {
            // Create a termination function
            KernelFunction terminateFunction = KernelFunctionFactory.CreateFromPrompt(
                $$$"""
                    Determine if the conversation is complete. If so, respond with a single word: yes.

                    History:

                    {{$history}}
                    """
                );

            // Create a selection function
            KernelFunction selectionFunction = KernelFunctionFactory.CreateFromPrompt(
                $$$"""
                    Your job is to determine which participant takes the next turn in a conversation according to the action of the most recent participant.
                    State only the name of the participant to take the next turn.

                    Choose only from these participants:
                    - {{{SocratesName}}}
                    - {{{PlatoName}}}
                    - {{{AristotleName}}}

                    Always follow these steps when selecting the next participant:
                    1) After user input, it is {{{SocratesName}}}'s turn to respond.
                    2) After {{{SocratesName}}} replies, it's {{{PlatoName}}}'s turn based on {{{SocratesName}}}'s response.
                    3) After {{{PlatoName}}} replies, it's {{{AristotleName}}}'s turn based on {{{SocratesName}}}'s response.
                    4) After {{{AristotleName}}} replies, it's {{{SocratesName}}}'s turn to summarize the responses and end the conversation.

                    Make sure each participant has a turn.

                    History:
                    {{$history}}
                    """
            );

            AgentGroupChat chat = new(socrates, aristotle, plato)
            {
                ExecutionSettings = new()
                {
                    TerminationStrategy = new KernelFunctionTerminationStrategy(terminateFunction, kernel)
                    {
                        Agents = [socrates],
                        ResultParser = (result) => result.GetValue<string>()?.Contains("yes", StringComparison.OrdinalIgnoreCase) ?? false,
                        HistoryVariableName = "history",
                        MaximumIterations = 4
                    },
                    SelectionStrategy = new KernelFunctionSelectionStrategy(selectionFunction, kernel)
                    {
                        AgentsVariableName = "agents",
                        HistoryVariableName = "history"
                    }
                }
            };

            chat.AddChatMessage(new ChatMessageContent(AuthorRole.User, prompt));

            await foreach (var content in chat.InvokeAsync())
            {
                Console.WriteLine();
                string color = content.AuthorName switch
                {
                    "Socrates" => "\u001b[34m", // Blue
                    "Aristotle" => "\u001b[32m", // Green
                    "Plato" => "\u001b[35m", // Magenta
                    _ => "\u001b[0m" // Default color
                };
                Console.WriteLine($"{color}[{content.AuthorName ?? "*"}]: '{content.Content}'\u001b[0m");
                Console.WriteLine();
            }

            Console.WriteLine($"# IS COMPLETE: {chat.IsComplete}");
        }
        finally
        {
            // Cleanup thread and vector-store
            await plato.DeleteThreadAsync(threadId);
            await plato.DeleteAsync();
            await vectorStoreClient.DeleteVectorStoreAsync(result.VectorStoreId);
            await fileClient.DeleteFileAsync(fileInfo.Id);
        }     
    }
}

