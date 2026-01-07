package dev.langchain4j.agentic.agent;

import static dev.langchain4j.agentic.internal.AgentUtil.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Executor;
import java.util.function.Function;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.agentic.declarative.TypedKey;
import dev.langchain4j.agentic.observability.AgentListener;
import dev.langchain4j.agentic.observability.ComposedAgentListener;
import dev.langchain4j.agentic.planner.AgentArgument;
import dev.langchain4j.agentic.scope.AgenticScope;
import dev.langchain4j.agentic.scope.DefaultAgenticScope;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.guardrail.InputGuardrail;
import dev.langchain4j.guardrail.OutputGuardrail;
import dev.langchain4j.guardrail.config.InputGuardrailsConfig;
import dev.langchain4j.guardrail.config.OutputGuardrailsConfig;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.ChatMemoryProvider;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.service.tool.ToolArgumentsErrorHandler;
import dev.langchain4j.service.tool.ToolExecutionErrorHandler;
import dev.langchain4j.service.tool.ToolExecutor;
import dev.langchain4j.service.tool.ToolProvider;

/**
 * Base class for agent builders containing all reflection-independent configuration.
 * This class contains all the state and configuration that doesn't depend on Java reflection,
 * annotations, or specific interface types.
 *
 * @param <T> The agent service interface type
 * @param <B> The concrete builder type (for fluent API pattern)
 */
public abstract class BaseAgentBuilder<T, B extends BaseAgentBuilder<T, B>> {

    // Agent metadata
    String name;
    String description;
    String outputKey;
    boolean async;
    List<AgentArgument> arguments;

    // Default values for agent arguments/keys
    private final Map<String, Object> defaultValues = new HashMap<>();

    // Chat model and memory configuration
    private ChatModel model;
    private ChatMemory chatMemory;
    private ChatMemoryProvider chatMemoryProvider;

    // Context and RAG configuration
    private Function<AgenticScope, String> contextProvider;
    private String[] contextProvidingAgents;
    private ContentRetriever contentRetriever;
    private RetrievalAugmentor retrievalAugmentor;
    private Function<Object, String> systemMessageProvider;

    // Guardrails configuration
    private InputGuardrailsConfig inputGuardrailsConfig;
    private OutputGuardrailsConfig outputGuardrailsConfig;
    private Class<? extends InputGuardrail>[] inputGuardrailClasses;
    private Class<? extends OutputGuardrail>[] outputGuardrailClasses;
    private InputGuardrail[] inputGuardrails;
    private OutputGuardrail[] outputGuardrails;

    // Tools configuration
    private Object[] objectsWithTools;
    private Map<ToolSpecification, ToolExecutor> toolsMap;
    private Set<String> immediateReturnToolNames;
    private ToolProvider toolProvider;
    private Integer maxSequentialToolsInvocations;
    private Function<ToolExecutionRequest, ToolExecutionResultMessage> hallucinatedToolNameStrategy;
    private boolean executeToolsConcurrently;
    private Executor concurrentToolsExecutor;
    private ToolArgumentsErrorHandler toolArgumentsErrorHandler;
    private ToolExecutionErrorHandler toolExecutionErrorHandler;

    // Observability
    AgentListener agentListener;

    /**
     * Returns the concrete builder instance for fluent API pattern.
     */
    @SuppressWarnings("unchecked")
    protected B self() {
        return (B) this;
    }

    // ===== Getters for protected access by subclasses =====

    protected Map<String, Object> getDefaultValues() {
        return defaultValues;
    }

    protected ChatModel getModel() {
        return model;
    }

    protected ChatMemory getChatMemory() {
        return chatMemory;
    }

    protected ChatMemoryProvider getChatMemoryProvider() {
        return chatMemoryProvider;
    }

    protected Function<AgenticScope, String> getContextProvider() {
        return contextProvider;
    }

    protected String[] getContextProvidingAgents() {
        return contextProvidingAgents;
    }

    protected ContentRetriever getContentRetriever() {
        return contentRetriever;
    }

    protected RetrievalAugmentor getRetrievalAugmentor() {
        return retrievalAugmentor;
    }

    protected Function<Object, String> getSystemMessageProvider() {
        return systemMessageProvider;
    }

    protected InputGuardrailsConfig getInputGuardrailsConfig() {
        return inputGuardrailsConfig;
    }

    protected OutputGuardrailsConfig getOutputGuardrailsConfig() {
        return outputGuardrailsConfig;
    }

    protected Class<? extends InputGuardrail>[] getInputGuardrailClasses() {
        return inputGuardrailClasses;
    }

    protected Class<? extends OutputGuardrail>[] getOutputGuardrailClasses() {
        return outputGuardrailClasses;
    }

    protected InputGuardrail[] getInputGuardrails() {
        return inputGuardrails;
    }

    protected OutputGuardrail[] getOutputGuardrails() {
        return outputGuardrails;
    }

    protected Object[] getObjectsWithTools() {
        return objectsWithTools;
    }

    protected Map<ToolSpecification, ToolExecutor> getToolsMap() {
        return toolsMap;
    }

    protected Set<String> getImmediateReturnToolNames() {
        return immediateReturnToolNames;
    }

    protected ToolProvider getToolProvider() {
        return toolProvider;
    }

    protected Integer getMaxSequentialToolsInvocations() {
        return maxSequentialToolsInvocations;
    }

    protected Function<ToolExecutionRequest, ToolExecutionResultMessage> getHallucinatedToolNameStrategy() {
        return hallucinatedToolNameStrategy;
    }

    protected boolean isExecuteToolsConcurrently() {
        return executeToolsConcurrently;
    }

    protected Executor getConcurrentToolsExecutor() {
        return concurrentToolsExecutor;
    }

    protected ToolArgumentsErrorHandler getToolArgumentsErrorHandler() {
        return toolArgumentsErrorHandler;
    }

    protected ToolExecutionErrorHandler getToolExecutionErrorHandler() {
        return toolExecutionErrorHandler;
    }

    // ===== Setters for fluent API (public) =====

    /**
     * Sets the chat model to use for this agent.
     */
    public B chatModel(ChatModel model) {
        this.model = model;
        return self();
    }

    /**
     * Sets the chat memory for this agent.
     */
    public B chatMemory(ChatMemory chatMemory) {
        this.chatMemory = chatMemory;
        return self();
    }

    /**
     * Sets the chat memory provider for this agent.
     */
    public B chatMemoryProvider(ChatMemoryProvider chatMemoryProvider) {
        this.chatMemoryProvider = chatMemoryProvider;
        return self();
    }

    /**
     * Checks if this agent has a non-default chat memory configuration.
     */
    boolean hasNonDefaultChatMemory() {
        return chatMemoryProvider != null;
    }

    /**
     * Configures tools from objects with @Tool-annotated methods.
     */
    public B tools(Object... objectsWithTools) {
        this.objectsWithTools = objectsWithTools;
        return self();
    }

    /**
     * Configures tools from a map of tool specifications to executors.
     */
    public B tools(Map<ToolSpecification, ToolExecutor> toolsMap) {
        this.toolsMap = toolsMap;
        return self();
    }

    /**
     * Configures tools from a map with immediate return tool names.
     */
    public B tools(Map<ToolSpecification, ToolExecutor> toolsMap, Set<String> immediateReturnToolNames) {
        this.toolsMap = toolsMap;
        this.immediateReturnToolNames = immediateReturnToolNames;
        return self();
    }

    /**
     * Sets a tool provider for dynamic tool provisioning.
     */
    public B toolProvider(ToolProvider toolProvider) {
        this.toolProvider = toolProvider;
        return self();
    }

    /**
     * Sets the maximum number of sequential tool invocations.
     */
    public B maxSequentialToolsInvocations(int maxSequentialToolsInvocations) {
        this.maxSequentialToolsInvocations = maxSequentialToolsInvocations;
        return self();
    }

    /**
     * Sets the strategy for handling hallucinated tool names.
     */
    public B hallucinatedToolNameStrategy(
            Function<ToolExecutionRequest, ToolExecutionResultMessage> hallucinatedToolNameStrategy) {
        this.hallucinatedToolNameStrategy = hallucinatedToolNameStrategy;
        return self();
    }

    /**
     * Sets the content retriever for RAG.
     */
    public B contentRetriever(ContentRetriever contentRetriever) {
        this.contentRetriever = contentRetriever;
        return self();
    }

    /**
     * Sets the retrieval augmentor for advanced RAG.
     */
    public B retrievalAugmentor(RetrievalAugmentor retrievalAugmentor) {
        this.retrievalAugmentor = retrievalAugmentor;
        return self();
    }

    /**
     * Sets the input guardrails configuration.
     */
    public B inputGuardrailsConfig(InputGuardrailsConfig inputGuardrailsConfig) {
        this.inputGuardrailsConfig = inputGuardrailsConfig;
        return self();
    }

    /**
     * Sets the output guardrails configuration.
     */
    public B outputGuardrailsConfig(OutputGuardrailsConfig outputGuardrailsConfig) {
        this.outputGuardrailsConfig = outputGuardrailsConfig;
        return self();
    }

    /**
     * Sets the input guardrail classes to instantiate.
     */
    public <I extends InputGuardrail> B inputGuardrailClasses(
            Class<? extends I>... inputGuardrailClasses) {
        this.inputGuardrailClasses = inputGuardrailClasses;
        return self();
    }

    /**
     * Sets the output guardrail classes to instantiate.
     */
    public <O extends OutputGuardrail> B outputGuardrailClasses(
            Class<? extends O>... outputGuardrailClasses) {
        this.outputGuardrailClasses = outputGuardrailClasses;
        return self();
    }

    /**
     * Sets the input guardrail instances.
     */
    public <I extends InputGuardrail> B inputGuardrails(I... inputGuardrails) {
        this.inputGuardrails = inputGuardrails;
        return self();
    }

    /**
     * Sets the output guardrail instances.
     */
    public <O extends OutputGuardrail> B outputGuardrails(O... outputGuardrails) {
        this.outputGuardrails = outputGuardrails;
        return self();
    }

    /**
     * Sets the agent name.
     */
    public B name(String name) {
        this.name = name;
        return self();
    }

    /**
     * Sets the agent description.
     */
    public B description(String description) {
        this.description = description;
        return self();
    }

    /**
     * Sets the output key (the key under which the result is stored in the agentic scope).
     */
    public B outputKey(String outputKey) {
        this.outputKey = outputKey;
        return self();
    }

    /**
     * Sets the output key using a typed key class.
     */
    public B outputKey(Class<? extends TypedKey<?>> outputKey) {
        return outputKey(keyName(outputKey));
    }

    /**
     * Sets whether this agent should execute asynchronously.
     */
    public B async(boolean async) {
        this.async = async;
        return self();
    }

    /**
     * Sets a custom context provider function.
     */
    public B context(Function<AgenticScope, String> contextProvider) {
        this.contextProvider = contextProvider;
        return self();
    }

    /**
     * Configures summarized context from other agents.
     */
    public B summarizedContext(String... contextProvidingAgents) {
        this.contextProvidingAgents = contextProvidingAgents;
        return self();
    }

    /**
     * Sets a custom system message provider.
     */
    public B systemMessageProvider(Function<Object, String> systemMessageProvider) {
        this.systemMessageProvider = systemMessageProvider;
        return self();
    }

    /**
     * Enables concurrent tool execution.
     */
    public B executeToolsConcurrently() {
        this.executeToolsConcurrently = true;
        return self();
    }

    /**
     * Enables concurrent tool execution with a custom executor.
     */
    public B executeToolsConcurrently(Executor executor) {
        this.executeToolsConcurrently = true;
        this.concurrentToolsExecutor = executor;
        return self();
    }

    /**
     * Sets the tool arguments error handler.
     */
    public B toolArgumentsErrorHandler(ToolArgumentsErrorHandler toolArgumentsErrorHandler) {
        this.toolArgumentsErrorHandler = toolArgumentsErrorHandler;
        return self();
    }

    /**
     * Sets the tool execution error handler.
     */
    public B toolExecutionErrorHandler(ToolExecutionErrorHandler toolExecutionErrorHandler) {
        this.toolExecutionErrorHandler = toolExecutionErrorHandler;
        return self();
    }

    /**
     * Sets a default value for a key in the agentic scope.
     */
    public B defaultKeyValue(String key, Object value) {
        this.defaultValues.put(key, value);
        return self();
    }

    /**
     * Sets a default value for a typed key in the agentic scope.
     */
    public <K> B defaultKeyValue(Class<? extends TypedKey<K>> key, K value) {
        return defaultKeyValue(keyName(key), value);
    }

    /**
     * Adds an agent listener for observability.
     */
    public B listener(AgentListener agentListener) {
        if (this.agentListener == null) {
            this.agentListener = agentListener;
        } else if (this.agentListener instanceof ComposedAgentListener composed) {
            composed.addListener(agentListener);
        } else {
            this.agentListener = new ComposedAgentListener(this.agentListener, agentListener);
        }
        return self();
    }

    /**
     * Gets the agent service interface class (reflection-dependent).
     * Subclasses that use reflection should override this method.
     */
    protected abstract Class<T> getAgentServiceClass();

    /**
     * Gets the agent return type (reflection-dependent).
     * Subclasses that use reflection should override this method.
     */
    protected abstract Class<?> getAgentReturnType();

    /**
     * Builds the agent instance.
     */
    public T build() {
        return build(null);
    }

    abstract T build(DefaultAgenticScope agenticScope);
}
