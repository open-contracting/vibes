/* global jQuery, Bloodhound, Handlebars, marked */

const searchProperties = ["code", "path", "title", "description"];

function isString(value) {
  return toString.call(value) === "[object String]";
}

function isObject(value) {
  return typeof value === "object" && value !== null;
}

function fields(metadata, filename, path, schema) {
  let data = [];
  if (Array.isArray(schema)) {
    for (const entry of schema) {
      data = data.concat(fields(metadata, filename, path, entry));
    }
  } else if (isObject(schema)) {
    const datum = {};
    if (isString(schema.title)) {
      datum.title = schema.title;
    }
    if (isString(schema.description)) {
      datum.description = new Handlebars.SafeString(marked(schema.description));
    }
    // If the schema has metadata properties.
    if (Object.keys(datum).length) {
      datum.extension = metadata;
      datum.schema = filename;
      datum.path = path;
      let types = [];
      if (isString(schema.type)) {
        types = [schema.type];
      } else if (schema.type) {
        types = schema.type;
      }
      const index = types.indexOf("null");
      if (index > -1) {
        types.splice(index, 1);
      }
      datum.type = types.join(", ");
      data.push(datum);
    }

    for (const property in schema) {
      let newPath;
      // Omit "definitions" and "properties" from the field's path. (Assumes "properties" is never a field name.)
      if ((property === "definitions" && path === "") || property === "properties") {
        newPath = path;
      } else if (path === "") {
        newPath = property;
      } else {
        newPath = `${path}.${property}`;
      }
      data = data.concat(fields(metadata, filename, newPath, schema[property]));
    }
  }
  return data;
}

const engine = new Bloodhound({
  datumTokenizer: (datum) => {
    let tokens = [];
    for (const property of searchProperties) {
      if (property in datum) {
        tokens = tokens.concat(Bloodhound.tokenizers.nonword(datum[property]));
        if (property === "code" || property === "path") {
          // Split on non-word characters, camel case and underscores.
          // `replace`` is used instead of `split`, because not all browsers implement lookbehind.
          tokens = tokens.concat(datum[property].replace(/([a-z])(?=[A-Z])/g, "$1 ").split(/[\W_]+/));
        }
      }
    }
    return tokens;
  },
  queryTokenizer: Bloodhound.tokenizers.nonword,
  prefetch: {
    url: "https://extensions.open-contracting.org/extensions.json",
    transform: (response) => {
      let data = [];
      for (const id in response) {
        const extension = response[id];
        const version = extension.versions[extension.latest_version];
        const schema = version.schemas["release-schema.json"];
        const metadata = {
          id: id,
          version: extension.latest_version,
          name: extension.name.en,
        };
        if (schema) {
          data = data.concat(fields(metadata, "release-schema.json", "", schema.en));
        }
        for (const codelist in version.codelists) {
          for (const row of version.codelists[codelist].en.rows) {
            data.push({
              extension: metadata,
              codelist: codelist,
              code: row.Code,
              title: row.Title,
              description: new Handlebars.SafeString(marked(row.Description || "")),
            });
          }
        }
      }
      return data;
    },
  },
});

engine.initialize().done(() => {
  jQuery("#typeahead").typeahead(
    {
      highlight: true,
    },
    {
      source: engine,
      limit: 1000,
      templates: {
        suggestion: Handlebars.compile(`
          <div class="panel panel-default">
            <div class="panel-heading">
              {{#if codelist}}
              Code:
              {{else}}
              Field:
              {{/if}}
              <strong>
                {{#if codelist}}
                {{code}}
                {{else}}
                {{path}}
                {{/if}}
              </strong>
              <span class="text-muted">
                {{#if type}}
                ({{type}})
                {{/if}}
                {{#if codelist}}
                in <a target="_blank" href="https://extensions.open-contracting.org/en/extensions/{{extension.id}}/{{extension.version}}/codelists/#{{codelist}}">{{codelist}}</a>
                {{/if}}
                in <a target="_blank" href="https://extensions.open-contracting.org/en/extensions/{{extension.id}}/{{extension.version}}/">{{extension.name}}</a>
              </span>
            </div>
            <div class="panel-body">
              <dl class="dl-horizontal">
                <dt>Title</dt>
                <dd>{{title}}</dd>
                <dt>Description</dt>
                <dd>{{description}}</dd>
              </dl>
            </div>
          </div>`),
      },
    },
  );
});
