import { createApp, defineComponent } from "vue";
import { createPinia } from "pinia";

export function withSetup<T>(composable: () => T): [T, ReturnType<typeof createApp>] {
  let result!: T;
  const app = createApp(
    defineComponent({
      setup() {
        result = composable();
        return () => null;
      },
    })
  );
  app.use(createPinia());
  app.mount(document.createElement("div"));
  return [result, app];
}
